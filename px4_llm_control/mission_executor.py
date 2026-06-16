#!/usr/bin/env python3
"""
Natural-language mission executor for PX4 multicopters.

Subscribes to `/nl_command` (std_msgs/String). Each message is one plain-English
instruction, e.g. "take off to 5 metres, fly 10 metres north, then hold for 3
seconds and land". An LLMPlanner (llm_planner.py) converts it into an ordered
list of mission steps — takeoff / goto / hold / velocity / attitude / land / rtl —
which this node executes one at a time over PX4 offboard control: the same
OffboardControlMode + TrajectorySetpoint + VehicleCommand pattern used by
interceptor_mission, plus VehicleAttitudeSetpoint for attitude steps.

Coordinate frame: NED, matching /fmu/out/vehicle_local_position_v1
(x = North, y = East, z = Down in metres; z < 0 is above the ground).

Status / progress is published on `/nl_mission/status` (std_msgs/String) for the
CLI (or any other listener) to print.
"""

import math
import threading
from collections import deque
from enum import Enum, auto
from queue import Empty, Queue

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data,
)

from sensor_msgs.msg import Image
from std_msgs.msg import String
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleAttitudeSetpoint, VehicleCommand,
    VehicleLocalPosition, VehicleStatus,
)
from ultralytics_ros.msg import YoloResult

from px4_llm_control.llm_planner import LLMPlanner, PlannerError

POS_TOLERANCE_M = 0.5    # metres — close enough to declare a goto/takeoff complete
YAW_TOLERANCE_RAD = 0.05 # ~3 degrees — close enough to declare a goto/takeoff heading reached
HEARTBEAT_TICKS = 15     # 1.5 s of setpoints before arm + offboard (matches interceptor_mission)
TICK_HZ         = 10.0

# Safety clamps applied to velocity/attitude steps regardless of what the planner returns.
MAX_VELOCITY_MPS   = 5.0    # vx/vy/vz clamp for 'velocity' steps
MAX_YAWSPEED_RADPS = 1.0    # yawspeed clamp for 'velocity' steps
MAX_TILT_RAD       = 0.35   # ~20 degrees — roll/pitch clamp for 'attitude' steps
MIN_THRUST         = 0.0
MAX_THRUST         = 0.9    # leave headroom below full throttle
MAX_TIMED_STEP_S   = 15.0   # duration clamp for 'hold' / 'velocity' / 'attitude' steps
EXTERNAL_WAIT_TIMEOUT_S = 10.0  # give up waiting for PX4 to auto-disarm after LAND/RTL

# 'follow' step: visual-servo control law over /yolo_result (ultralytics_ros).
CAMERA_WIDTH  = 640    # matches Tools/simulation/gz/models/mono_cam/model.sdf (x500_mono_cam)
CAMERA_HEIGHT = 480
FOLLOW_TARGET_BBOX_HEIGHT_FRAC = 0.25  # desired bbox height / image height ("follow distance") —
                                       # smaller = stand farther back, giving more margin before
                                       # the target's apparent angular speed outruns yaw tracking
FOLLOW_KP_YAW             = 1.0   # rad/s yawspeed per unit normalized horizontal error
FOLLOW_YAW_DEADBAND       = 0.05  # |err_x| below this -> yawspeed = 0 (avoids hunting/overshoot
                                  # oscillation once the target is roughly centered)
FOLLOW_KP_FORWARD         = 6.0   # m/s forward speed per unit normalized size error — sized so
                                  # FOLLOW_KP_FORWARD * FOLLOW_TARGET_BBOX_HEIGHT_FRAC ==
                                  # FOLLOW_MAX_SPEED_MPS, i.e. full speed is reachable when the
                                  # target shrinks to ~nothing (far away), not just when it's close
FOLLOW_FORWARD_DEADBAND  = 0.02  # |err_size| below this -> forward_speed = 0 (avoids
                                  # forward/back "breathing" from the higher gain above)
FOLLOW_MAX_SPEED_MPS      = 1.5
FOLLOW_MAX_YAWSPEED_RADPS = 0.6
FOLLOW_LOST_TIMEOUT_S     = 3.0   # hover this long after losing the target before dropping the lock
                                  # (follow keeps hovering and resumes tracking if the target reappears)
FOLLOW_REIDENTIFY_INTERVAL_S = 1.0   # how often to re-run Claude-vision target selection


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """ZYX Euler angles (radians) -> quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


class State(Enum):
    GROUNDED      = auto()   # disarmed on the ground, streaming heartbeat setpoints —
                              # arms + engages offboard once a step is queued
    IDLE          = auto()   # holding position, waiting for the next mission step
    TAKEOFF       = auto()   # climbing to a commanded altitude
    GOTO          = auto()   # flying to an (x, y, z, yaw) setpoint
    HOLD          = auto()   # holding position for a fixed duration
    VELOCITY      = auto()   # commanding an NED velocity (+ optional yaw rate) for a fixed duration
    ATTITUDE      = auto()   # commanding a roll/pitch/yaw + thrust setpoint for a fixed duration
    FOLLOW        = auto()   # visually tracking a detected object via /yolo_result, until interrupted or lost
    LAND          = auto()   # one-shot: hand off to PX4's AUTO_LAND
    RTL           = auto()   # one-shot: hand off to PX4's AUTO_RTL
    EXTERNAL_WAIT = auto()   # PX4-driven land/RTL in progress — wait for disarm


class MissionExecutor(Node):

    def __init__(self):
        super().__init__('nl_mission_executor')

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub_ocm = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', px4_qos)
        self._pub_tsp = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', px4_qos)
        self._pub_cmd = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', px4_qos)
        self._pub_att = self.create_publisher(
            VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint_v1', px4_qos)
        self._pub_status = self.create_publisher(String, '/nl_mission/status', 10)

        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._cb_pos, px4_qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1', self._cb_status, px4_qos)
        self.create_subscription(String, '/nl_command', self._cb_command, 10)
        self.create_subscription(YoloResult, '/yolo_result', self._cb_yolo, 10)
        self.create_subscription(
            Image, '/camera/color/image_raw', self._cb_image, qos_profile_sensor_data)

        self._pos    = VehicleLocalPosition()
        self._status = VehicleStatus()
        self._last_arming_state = None
        self._last_nav_state    = None

        self._state    = State.GROUNDED
        self._hb_count = 0

        # Fixed setpoint the drone holds while idle / mid-hold (avoids feeding back
        # the noisy live position estimate as its own setpoint, which would drift).
        self._hold_x = 0.0
        self._hold_y = 0.0
        self._hold_z = 0.0
        self._hold_yaw = None

        self._steps       = deque()                 # pending mission steps (dicts)
        self._goto_target = (0.0, 0.0, 0.0, None)   # (x, y, z, yaw) for TAKEOFF / GOTO
        self._velocity_target = (0.0, 0.0, 0.0, None)  # (vx, vy, vz, yawspeed) for VELOCITY
        self._attitude_target = (0.0, 0.0, 0.0, 0.0)   # (roll, pitch, yaw, thrust) for ATTITUDE
        self._timer_until = 0.0                      # clock seconds for HOLD / VELOCITY / ATTITUDE / EXTERNAL_WAIT

        # 'follow' state
        self._cv_bridge = CvBridge()
        self._latest_frame = None         # most recent /camera/color/image_raw frame (BGR numpy array)
        self._latest_detections = []      # most recent vision_msgs/Detection2D list from /yolo_result
        self._follow_target_class = None  # lowercase COCO class name being followed
        self._follow_description = None   # optional free-text description for Claude-vision target selection
        self._follow_last_bbox = None     # (x, y) pixel center of the previously-locked detection
        self._follow_lost_since = None    # wall clock when the target was last seen, or None
        self._follow_locked = False       # whether we've reported "tracking" since FOLLOW started
        self._follow_vision_pending = False    # whether a Claude-vision identification request is in flight
        self._follow_last_vision_time = None   # wall clock of the last vision identification request

        # The LLM call blocks on the network — run it on a worker thread so the
        # 10 Hz offboard heartbeat (required to stay in OFFBOARD mode) never stalls.
        self._planner  = LLMPlanner()
        self._plan_in  = Queue()   # (instruction, state-snapshot) → worker thread
        self._plan_out = Queue()   # ('ok'|'error', instruction, payload) → tick thread
        threading.Thread(target=self._planner_worker, daemon=True).start()

        # Claude-vision target selection for "follow ... <description>" — also blocks
        # on the network, so it gets its own worker thread.
        self._vision_in  = Queue()   # (jpeg_bytes, description, candidate_centers) → worker thread
        self._vision_out = Queue()   # ('ok'|'error', payload, candidate_centers, description) → tick thread
        threading.Thread(target=self._vision_worker, daemon=True).start()

        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info('nl_mission_executor ready — send instructions on /nl_command')

    # ── telemetry ─────────────────────────────────────────────────────────────

    def _cb_pos(self, msg: VehicleLocalPosition):
        self._pos = msg

    def _cb_status(self, msg: VehicleStatus):
        self._status = msg
        if msg.arming_state != self._last_arming_state or msg.nav_state != self._last_nav_state:
            self._status_msg(f'PX4: arming_state={msg.arming_state}, nav_state={msg.nav_state}')
            self._last_arming_state = msg.arming_state
            self._last_nav_state    = msg.nav_state

    def _cb_yolo(self, msg: YoloResult):
        self._latest_detections = msg.detections.detections

    def _cb_image(self, msg: Image):
        self._latest_frame = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _cb_command(self, msg: String):
        instruction = msg.data.strip()
        if not instruction:
            return
        snapshot = {
            'x': self._pos.x, 'y': self._pos.y, 'z': self._pos.z,
            'heading': self._pos.heading,
        }
        self._status_msg(f'planning: "{instruction}"')
        self._plan_in.put((instruction, snapshot))

    # ── LLM worker thread ─────────────────────────────────────────────────────

    def _planner_worker(self):
        while True:
            instruction, snapshot = self._plan_in.get()
            try:
                steps = self._planner.plan(instruction, snapshot)
                self._plan_out.put(('ok', instruction, steps))
            except (PlannerError, Exception) as exc:   # noqa: BLE001 — surface SDK/network errors too
                self._plan_out.put(('error', instruction, str(exc)))

    def _drain_plans(self):
        while True:
            try:
                kind, instruction, payload = self._plan_out.get_nowait()
            except Empty:
                return
            if kind == 'ok':
                self._steps.extend(payload)
                actions = ', '.join(step['action'] for step in payload)
                self._status_msg(f'queued {len(payload)} step(s) for "{instruction}": {actions}')
            else:
                self._status_msg(f'planning failed for "{instruction}": {payload}')

    # ── vision worker thread ─────────────────────────────────────────────────

    def _vision_worker(self):
        while True:
            jpeg_bytes, description, candidate_centers = self._vision_in.get()
            try:
                index = self._planner.identify_target(jpeg_bytes, description, len(candidate_centers))
                self._vision_out.put(('ok', index, candidate_centers, description))
            except Exception as exc:   # noqa: BLE001 — surface SDK/network errors too
                self._vision_out.put(('error', str(exc), candidate_centers, description))

    def _drain_vision(self):
        while True:
            try:
                kind, payload, candidate_centers, description = self._vision_out.get_nowait()
            except Empty:
                return
            self._follow_vision_pending = False
            if kind == 'ok':
                if payload is not None and 1 <= payload <= len(candidate_centers):
                    self._follow_last_bbox = candidate_centers[payload - 1]
                    self._status_msg(
                        f'follow: vision matched "{description}" to candidate {payload}/{len(candidate_centers)}')
                else:
                    self._status_msg(
                        f'follow: vision found no candidate matching "{description}" — keeping current target')
            else:
                self._status_msg(f'follow: vision identification error: {payload}')

    # ── helpers ───────────────────────────────────────────────────────────────

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _ts(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _status_msg(self, text: str):
        self.get_logger().info(text)
        self._pub_status.publish(String(data=text))

    def _send_ocm(self):
        msg = OffboardControlMode()
        msg.position  = self._state not in (State.VELOCITY, State.ATTITUDE)
        msg.velocity  = self._state in (State.VELOCITY, State.FOLLOW)
        msg.attitude  = self._state == State.ATTITUDE
        msg.timestamp = self._ts()
        self._pub_ocm.publish(msg)

    def _send_setpoint(self, x: float, y: float, z: float, yaw=None):
        msg = TrajectorySetpoint()
        msg.position  = [float(x), float(y), float(z)]
        msg.yaw       = float('nan') if yaw is None else float(yaw)
        msg.timestamp = self._ts()
        self._pub_tsp.publish(msg)

    def _send_velocity_setpoint(self, vx: float, vy: float, vz: float, yawspeed=None, hold_z=None):
        # hold_z: if given, z is position-held at this altitude (mixed
        # position/velocity setpoint) instead of velocity-controlled — pure
        # velocity-mode vz=0 has a small steady-state descent rate on PX4, which
        # is fine for short timed VELOCITY/move segments but causes FOLLOW (which
        # runs indefinitely) to slowly sink until it hits the ground.
        msg = TrajectorySetpoint()
        nan = float('nan')
        msg.position  = [nan, nan, nan if hold_z is None else float(hold_z)]
        msg.velocity  = [float(vx), float(vy), nan if hold_z is not None else float(vz)]
        msg.yaw       = nan
        msg.yawspeed  = nan if yawspeed is None else float(yawspeed)
        msg.timestamp = self._ts()
        self._pub_tsp.publish(msg)

    def _send_attitude_setpoint(self, roll: float, pitch: float, yaw: float, thrust: float):
        msg = VehicleAttitudeSetpoint()
        msg.q_d        = euler_to_quaternion(roll, pitch, yaw)
        msg.thrust_body = [0.0, 0.0, -float(thrust)]
        msg.timestamp  = self._ts()
        self._pub_att.publish(msg)

    def _send_cmd(self, command: int, **kw):
        msg = VehicleCommand()
        msg.command          = command
        msg.param1           = float(kw.get('p1', 0))
        msg.param2           = float(kw.get('p2', 0))
        msg.param3           = float(kw.get('p3', 0))
        msg.param4           = float(kw.get('p4', 0))
        msg.param5           = float(kw.get('p5', 0))
        msg.param6           = float(kw.get('p6', 0))
        msg.param7           = float(kw.get('p7', 0))
        msg.target_system    = 1
        msg.target_component = 1
        msg.source_system    = 1
        msg.source_component = 1
        msg.from_external    = True
        msg.timestamp        = self._ts()
        self._pub_cmd.publish(msg)

    def _arm(self):
        # p2=21196.0 forces arm in SITL regardless of pre-flight check failures
        self._send_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, p1=1.0, p2=21196.0)

    def _engage_offboard(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, p1=1.0, p2=6.0)

    def _at_position(self, x: float, y: float, z: float, tol: float = POS_TOLERANCE_M) -> bool:
        return math.dist((self._pos.x, self._pos.y, self._pos.z), (x, y, z)) < tol

    def _at_yaw(self, yaw, tol: float = YAW_TOLERANCE_RAD) -> bool:
        if yaw is None:
            return True
        error = (yaw - self._pos.heading + math.pi) % (2 * math.pi) - math.pi
        return abs(error) < tol

    def _transition(self, new_state: State):
        self.get_logger().info(f'{self._state.name} → {new_state.name}')
        self._state = new_state

    # ── state machine ─────────────────────────────────────────────────────────

    def _tick(self):
        self._drain_plans()
        self._drain_vision()
        self._send_ocm()

        if   self._state == State.GROUNDED:      self._s_grounded()
        elif self._state == State.IDLE:          self._s_idle()
        elif self._state == State.TAKEOFF:       self._s_takeoff()
        elif self._state == State.GOTO:          self._s_goto()
        elif self._state == State.HOLD:          self._s_hold()
        elif self._state == State.VELOCITY:      self._s_velocity()
        elif self._state == State.ATTITUDE:      self._s_attitude()
        elif self._state == State.FOLLOW:        self._s_follow()
        elif self._state == State.LAND:          self._s_land()
        elif self._state == State.RTL:           self._s_rtl()
        elif self._state == State.EXTERNAL_WAIT: self._s_external_wait()

    def _s_grounded(self):
        self._send_setpoint(self._pos.x, self._pos.y, self._pos.z, yaw=self._pos.heading)
        if self._hb_count < HEARTBEAT_TICKS:
            self._hb_count += 1
            return
        if self._steps:
            self._hold_x, self._hold_y, self._hold_z = self._pos.x, self._pos.y, self._pos.z
            self._status_msg('arming and engaging offboard')
            self._engage_offboard()
            self._arm()
            self._transition(State.IDLE)

    def _s_idle(self):
        self._send_setpoint(self._hold_x, self._hold_y, self._hold_z, yaw=self._hold_yaw)
        if self._steps:
            self._dispatch_next_step()

    def _dispatch_next_step(self):
        step = self._steps.popleft()
        action = step['action']
        self._status_msg(f'executing: {step}')

        if action == 'takeoff':
            self._goto_target = (self._pos.x, self._pos.y, -abs(step['altitude']), self._pos.heading)
            self._transition(State.TAKEOFF)
        elif action == 'goto':
            self._goto_target = (step['x'], step['y'], step['z'], step.get('yaw'))
            self._transition(State.GOTO)
        elif action == 'move':
            forward, right, dz, yaw_delta = (
                step['forward'], step['right'], step['dz'], step['yaw_delta'])

            if yaw_delta != 0.0 and (forward != 0.0 or right != 0.0 or dz != 0.0):
                # Rotate in place first, then translate using the post-rotation
                # heading: split into two GOTO targets — _at_yaw makes the
                # rotation finish before the translation starts.
                self._steps.appendleft({
                    'action': 'move', 'forward': forward, 'right': right, 'dz': dz, 'yaw_delta': 0.0,
                })
                forward = right = dz = 0.0

            heading = self._pos.heading + yaw_delta
            dx = forward * math.cos(heading) - right * math.sin(heading)
            dy = forward * math.sin(heading) + right * math.cos(heading)
            target_yaw = heading if yaw_delta != 0.0 else None
            self._goto_target = (self._pos.x + dx, self._pos.y + dy, self._pos.z + dz, target_yaw)
            self._transition(State.GOTO)
        elif action == 'hold':
            self._timer_until = self._now_s() + _clamp(float(step['duration']), 0.0, MAX_TIMED_STEP_S)
            self._transition(State.HOLD)
        elif action == 'velocity':
            vx, vy = step['vx'], step['vy']
            forward_speed, right_speed = step['forward_speed'], step['right_speed']
            if forward_speed != 0.0 or right_speed != 0.0:
                # Convert body-frame speeds using the real heading at dispatch time —
                # same formula as the 'move' action — so the LLM never does trig.
                heading = self._pos.heading
                vx += forward_speed * math.cos(heading) - right_speed * math.sin(heading)
                vy += forward_speed * math.sin(heading) + right_speed * math.cos(heading)
            self._velocity_target = (
                _clamp(vx, -MAX_VELOCITY_MPS, MAX_VELOCITY_MPS),
                _clamp(vy, -MAX_VELOCITY_MPS, MAX_VELOCITY_MPS),
                _clamp(step['vz'], -MAX_VELOCITY_MPS, MAX_VELOCITY_MPS),
                None if step.get('yawspeed') is None
                    else _clamp(step['yawspeed'], -MAX_YAWSPEED_RADPS, MAX_YAWSPEED_RADPS),
            )
            self._timer_until = self._now_s() + _clamp(float(step['duration']), 0.0, MAX_TIMED_STEP_S)
            self._transition(State.VELOCITY)
        elif action == 'attitude':
            self._attitude_target = (
                _clamp(step['roll'], -MAX_TILT_RAD, MAX_TILT_RAD),
                _clamp(step['pitch'], -MAX_TILT_RAD, MAX_TILT_RAD),
                step['yaw'],
                _clamp(step['thrust'], MIN_THRUST, MAX_THRUST),
            )
            self._timer_until = self._now_s() + _clamp(float(step['duration']), 0.0, MAX_TIMED_STEP_S)
            self._transition(State.ATTITUDE)
        elif action == 'follow':
            self._follow_target_class = step['target'].strip().lower()
            self._follow_description = step.get('description')
            self._follow_last_bbox = None
            self._follow_lost_since = None
            self._follow_locked = False
            self._follow_vision_pending = False
            self._follow_last_vision_time = None
            self._transition(State.FOLLOW)
        elif action == 'land':
            self._transition(State.LAND)
        elif action == 'rtl':
            self._transition(State.RTL)

    def _s_takeoff(self):
        x, y, z, yaw = self._goto_target
        self._send_setpoint(x, y, z, yaw=yaw)
        if self._at_position(x, y, z, tol=0.3) and self._at_yaw(yaw):
            self._hold_x, self._hold_y, self._hold_z = x, y, z
            self._hold_yaw = yaw
            self._status_msg('takeoff complete')
            self._transition(State.IDLE)

    def _s_goto(self):
        x, y, z, yaw = self._goto_target
        self._send_setpoint(x, y, z, yaw=yaw)
        if self._at_position(x, y, z) and self._at_yaw(yaw):
            self._hold_x, self._hold_y, self._hold_z = x, y, z
            if yaw is not None:
                self._hold_yaw = yaw
            self._status_msg(f'reached ({x:.1f}, {y:.1f}, {z:.1f})')
            self._transition(State.IDLE)

    def _s_hold(self):
        self._send_setpoint(self._hold_x, self._hold_y, self._hold_z)
        if self._now_s() >= self._timer_until:
            self._transition(State.IDLE)

    def _s_velocity(self):
        vx, vy, vz, yawspeed = self._velocity_target
        self._send_velocity_setpoint(vx, vy, vz, yawspeed)
        if self._now_s() >= self._timer_until:
            self._hold_x, self._hold_y, self._hold_z = self._pos.x, self._pos.y, self._pos.z
            self._hold_yaw = self._pos.heading
            self._status_msg(
                f'velocity segment complete at ({self._hold_x:.1f}, {self._hold_y:.1f}, {self._hold_z:.1f})')
            self._transition(State.IDLE)

    def _s_attitude(self):
        roll, pitch, yaw, thrust = self._attitude_target
        self._send_attitude_setpoint(roll, pitch, yaw, thrust)
        if self._now_s() >= self._timer_until:
            self._hold_x, self._hold_y, self._hold_z = self._pos.x, self._pos.y, self._pos.z
            self._hold_yaw = self._pos.heading
            self._status_msg('attitude segment complete')
            self._transition(State.IDLE)

    def _find_follow_target(self):
        matches = [
            d for d in self._latest_detections
            if d.results and d.results[0].hypothesis.class_id.lower() == self._follow_target_class
        ]
        if not matches:
            return None
        if self._follow_last_bbox is not None:
            lx, ly = self._follow_last_bbox
            return min(
                matches,
                key=lambda d: (d.bbox.center.position.x - lx) ** 2
                             + (d.bbox.center.position.y - ly) ** 2,
            )
        return max(matches, key=lambda d: d.results[0].hypothesis.score)

    def _request_target_identification(self):
        """Ask Claude vision which detected `_follow_target_class` instance matches
        `_follow_description`, by drawing numbered boxes on the latest camera frame
        and sending it off on the vision worker thread (non-blocking)."""
        if self._latest_frame is None:
            return
        candidates = [
            d for d in self._latest_detections
            if d.results and d.results[0].hypothesis.class_id.lower() == self._follow_target_class
        ]
        if not candidates:
            return

        frame = self._latest_frame.copy()
        candidate_centers = []
        for i, d in enumerate(candidates, start=1):
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            sx, sy = d.bbox.size_x, d.bbox.size_y
            x1, y1 = int(cx - sx / 2), int(cy - sy / 2)
            x2, y2 = int(cx + sx / 2), int(cy + sy / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(i), (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            candidate_centers.append((cx, cy))

        ok, jpeg = cv2.imencode('.jpg', frame)
        if not ok:
            return

        self._follow_vision_pending = True
        self._follow_last_vision_time = self._now_s()
        self._vision_in.put((jpeg.tobytes(), self._follow_description, candidate_centers))

    def _s_follow(self):
        # Yield to any newly queued instruction (e.g. "stop", "land", a new "follow ...").
        if self._steps:
            self._hold_x, self._hold_y, self._hold_z = self._pos.x, self._pos.y, self._pos.z
            self._hold_yaw = self._pos.heading
            self._transition(State.IDLE)
            return

        if self._follow_description and not self._follow_vision_pending and (
            self._follow_last_vision_time is None
            or self._now_s() - self._follow_last_vision_time >= FOLLOW_REIDENTIFY_INTERVAL_S
        ):
            self._request_target_identification()

        target = self._find_follow_target()
        if target is None:
            self._send_velocity_setpoint(0.0, 0.0, 0.0, yawspeed=0.0, hold_z=self._hold_z)
            if self._follow_lost_since is None:
                self._follow_lost_since = self._now_s()
            elif self._follow_locked and self._now_s() - self._follow_lost_since > FOLLOW_LOST_TIMEOUT_S:
                # Drop the lock but keep hovering in FOLLOW — if the target reappears,
                # _find_follow_target falls back to its highest-confidence match (since
                # _follow_last_bbox is cleared) and tracking resumes automatically.
                self._follow_last_bbox = None
                self._follow_locked = False
                self._status_msg(f'follow: lost "{self._follow_target_class}" — waiting for it to reappear')
            return

        if not self._follow_locked:
            self._follow_locked = True
            self._status_msg(f'follow: tracking "{self._follow_target_class}"')
        self._follow_lost_since = None

        cx, cy = target.bbox.center.position.x, target.bbox.center.position.y
        self._follow_last_bbox = (cx, cy)

        err_x    = (cx - CAMERA_WIDTH / 2) / (CAMERA_WIDTH / 2)
        err_size = (FOLLOW_TARGET_BBOX_HEIGHT_FRAC * CAMERA_HEIGHT - target.bbox.size_y) / CAMERA_HEIGHT

        if abs(err_x) < FOLLOW_YAW_DEADBAND:
            yawspeed = 0.0
        else:
            yawspeed = _clamp(FOLLOW_KP_YAW * err_x, -FOLLOW_MAX_YAWSPEED_RADPS, FOLLOW_MAX_YAWSPEED_RADPS)
        if abs(err_size) < FOLLOW_FORWARD_DEADBAND:
            forward_speed = 0.0
        else:
            forward_speed = _clamp(FOLLOW_KP_FORWARD * err_size, -FOLLOW_MAX_SPEED_MPS, FOLLOW_MAX_SPEED_MPS)

        heading = self._pos.heading
        vx = forward_speed * math.cos(heading)
        vy = forward_speed * math.sin(heading)
        self._send_velocity_setpoint(vx, vy, 0.0, yawspeed=yawspeed, hold_z=self._hold_z)

    def _s_land(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self._status_msg('landing')
        self._timer_until = self._now_s() + EXTERNAL_WAIT_TIMEOUT_S
        self._transition(State.EXTERNAL_WAIT)

    def _s_rtl(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self._status_msg('returning to launch')
        self._timer_until = self._now_s() + EXTERNAL_WAIT_TIMEOUT_S
        self._transition(State.EXTERNAL_WAIT)

    def _s_external_wait(self):
        # PX4 is flying its own LAND/RTL sequence on its own — wait for it to
        # disarm, then go back to GROUNDED so the next instruction re-arms and
        # re-engages offboard on demand. If PX4 doesn't report disarmed within
        # EXTERNAL_WAIT_TIMEOUT_S (e.g. auto-disarm-after-land didn't fire), give up
        # waiting anyway so new instructions aren't stuck in _steps forever.
        if self._status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
            self._steps.clear()
            self._status_msg('landed and disarmed — send another instruction when ready')
            self._hb_count = 0
            self._transition(State.GROUNDED)
        elif self._now_s() >= self._timer_until:
            self._status_msg('did not see disarm after land/RTL — resuming control anyway')
            self._hb_count = 0
            self._transition(State.GROUNDED)


def main(args=None):
    rclpy.init(args=args)
    node = MissionExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
