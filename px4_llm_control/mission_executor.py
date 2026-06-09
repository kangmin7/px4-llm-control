#!/usr/bin/env python3
"""
Natural-language mission executor for PX4 multicopters.

Subscribes to `/nl_command` (std_msgs/String). Each message is one plain-English
instruction, e.g. "take off to 5 metres, fly 10 metres north, then hold for 3
seconds and land". An LLMPlanner (llm_planner.py) converts it into an ordered
list of mission steps — takeoff / goto / hold / land / rtl — which this node
executes one at a time over PX4 offboard control: the same OffboardControlMode +
TrajectorySetpoint + VehicleCommand pattern used by interceptor_mission.

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

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy,
)

from std_msgs.msg import String
from px4_msgs.msg import (
    OffboardControlMode, TrajectorySetpoint, VehicleCommand,
    VehicleLocalPosition, VehicleStatus,
)

from px4_llm_control.llm_planner import LLMPlanner, PlannerError

POS_TOLERANCE_M = 0.5    # metres — close enough to declare a goto/takeoff complete
HEARTBEAT_TICKS = 15     # 1.5 s of setpoints before arm + offboard (matches interceptor_mission)
TICK_HZ         = 10.0


class State(Enum):
    INIT          = auto()   # pre-arm heartbeat, then arm + engage offboard
    IDLE          = auto()   # holding position, waiting for the next mission step
    TAKEOFF       = auto()   # climbing to a commanded altitude
    GOTO          = auto()   # flying to an (x, y, z, yaw) setpoint
    HOLD          = auto()   # holding position for a fixed duration
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
        self._pub_status = self.create_publisher(String, '/nl_mission/status', 10)

        self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position_v1', self._cb_pos, px4_qos)
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v4', self._cb_status, px4_qos)
        self.create_subscription(String, '/nl_command', self._cb_command, 10)

        self._pos    = VehicleLocalPosition()
        self._status = VehicleStatus()

        self._state    = State.INIT
        self._hb_count = 0

        # Fixed setpoint the drone holds while idle / mid-hold (avoids feeding back
        # the noisy live position estimate as its own setpoint, which would drift).
        self._hold_x = 0.0
        self._hold_y = 0.0
        self._hold_z = 0.0

        self._steps       = deque()                 # pending mission steps (dicts)
        self._goto_target = (0.0, 0.0, 0.0, None)   # (x, y, z, yaw) for TAKEOFF / GOTO
        self._hold_until  = 0.0                      # clock seconds for HOLD

        # The LLM call blocks on the network — run it on a worker thread so the
        # 10 Hz offboard heartbeat (required to stay in OFFBOARD mode) never stalls.
        self._planner  = LLMPlanner()
        self._plan_in  = Queue()   # (instruction, state-snapshot) → worker thread
        self._plan_out = Queue()   # ('ok'|'error', instruction, payload) → tick thread
        threading.Thread(target=self._planner_worker, daemon=True).start()

        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info('nl_mission_executor ready — send instructions on /nl_command')

    # ── telemetry ─────────────────────────────────────────────────────────────

    def _cb_pos(self, msg: VehicleLocalPosition):
        self._pos = msg

    def _cb_status(self, msg: VehicleStatus):
        self._status = msg

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
        msg.position  = True
        msg.timestamp = self._ts()
        self._pub_ocm.publish(msg)

    def _send_setpoint(self, x: float, y: float, z: float, yaw=None):
        msg = TrajectorySetpoint()
        msg.position  = [float(x), float(y), float(z)]
        msg.yaw       = float('nan') if yaw is None else float(yaw)
        msg.timestamp = self._ts()
        self._pub_tsp.publish(msg)

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

    def _transition(self, new_state: State):
        self.get_logger().info(f'{self._state.name} → {new_state.name}')
        self._state = new_state

    # ── state machine ─────────────────────────────────────────────────────────

    def _tick(self):
        self._drain_plans()
        self._send_ocm()

        if   self._state == State.INIT:          self._s_init()
        elif self._state == State.IDLE:          self._s_idle()
        elif self._state == State.TAKEOFF:       self._s_takeoff()
        elif self._state == State.GOTO:          self._s_goto()
        elif self._state == State.HOLD:          self._s_hold()
        elif self._state == State.LAND:          self._s_land()
        elif self._state == State.RTL:           self._s_rtl()
        elif self._state == State.EXTERNAL_WAIT: self._s_external_wait()

    def _s_init(self):
        self._send_setpoint(self._pos.x, self._pos.y, self._pos.z, yaw=self._pos.heading)
        self._hb_count += 1
        if self._hb_count >= HEARTBEAT_TICKS:
            self._hold_x, self._hold_y, self._hold_z = self._pos.x, self._pos.y, self._pos.z
            self._engage_offboard()
            self._arm()
            self._transition(State.IDLE)

    def _s_idle(self):
        self._send_setpoint(self._hold_x, self._hold_y, self._hold_z)
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
        elif action == 'hold':
            self._hold_until = self._now_s() + float(step['duration'])
            self._transition(State.HOLD)
        elif action == 'land':
            self._transition(State.LAND)
        elif action == 'rtl':
            self._transition(State.RTL)

    def _s_takeoff(self):
        x, y, z, yaw = self._goto_target
        self._send_setpoint(x, y, z, yaw=yaw)
        if self._at_position(x, y, z, tol=0.3):
            self._hold_x, self._hold_y, self._hold_z = x, y, z
            self._status_msg('takeoff complete')
            self._transition(State.IDLE)

    def _s_goto(self):
        x, y, z, yaw = self._goto_target
        self._send_setpoint(x, y, z, yaw=yaw)
        if self._at_position(x, y, z):
            self._hold_x, self._hold_y, self._hold_z = x, y, z
            self._status_msg(f'reached ({x:.1f}, {y:.1f}, {z:.1f})')
            self._transition(State.IDLE)

    def _s_hold(self):
        self._send_setpoint(self._hold_x, self._hold_y, self._hold_z)
        if self._now_s() >= self._hold_until:
            self._transition(State.IDLE)

    def _s_land(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self._status_msg('landing')
        self._transition(State.EXTERNAL_WAIT)

    def _s_rtl(self):
        self._send_cmd(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
        self._status_msg('returning to launch')
        self._transition(State.EXTERNAL_WAIT)

    def _s_external_wait(self):
        # PX4 is flying its own LAND/RTL sequence on its own — wait for it to
        # disarm, then re-arm and re-engage offboard so the next instruction
        # (if any was queued) can run.
        if self._status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
            self._steps.clear()
            self._status_msg('landed and disarmed — ready for the next instruction')
            self._hb_count = 0
            self._transition(State.INIT)


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
