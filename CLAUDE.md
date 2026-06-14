# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`px4_llm_control` is a ROS 2 (ament_python) package that lets a user control a PX4
multicopter with plain-English instructions. An LLM (Claude, via the Anthropic API)
converts each instruction into a list of structured mission steps, which a state-machine
node executes over PX4 offboard control via uXRCE-DDS.

This package lives in a colcon workspace at `~/ros2_ws`; this directory is `~/ros2_ws/src/px4_llm_control`.

## Build & run

```bash
# Build (from the workspace root)
cd ~/ros2_ws
colcon build --packages-select px4_llm_control
source install/setup.bash
```

Running the full system requires three things, in order:

```bash
# 1. PX4 SITL (x500_mono_cam adds a forward camera, used by "follow" steps;
#    otherwise identical flight dynamics to plain x500)
cd ~/PX4-Autopilot
make px4_sitl gz_x500_mono_cam

# 2. uXRCE-DDS bridge
MicroXRCEAgent udp4 -p 8888

# 3. LLM mission control (executor + GUI)
export ANTHROPIC_API_KEY=your_api_key_here
ros2 launch px4_llm_control px4_llm_control.launch.py
```

`command_gui` (Tkinter) and `command_cli` (stdin) are alternative front ends — both just
publish `/nl_command` strings and print `/nl_mission/status`. The launch file starts the
GUI; run `ros2 run px4_llm_control command_cli` instead/additionally for a terminal interface.

`NL_MISSION_MODEL` overrides the Claude model used by the planner (default `claude-sonnet-4-6`).

For "follow" instructions, also run the vision pipeline (camera bridge + YOLO tracker):

```bash
# 4. Vision pipeline (camera bridge + ultralytics_ros tracker -> /yolo_result)
ros2 launch px4_llm_control vision.launch.py
```

This is optional — without it, "follow" steps just report a lost target after
`FOLLOW_LOST_TIMEOUT_S` and return to `IDLE`.

## Testing & linting

`package.xml` declares the standard ament test deps (`ament_copyright`, `ament_flake8`,
`ament_pep257`, `python3-pytest`), run via:

```bash
colcon test --packages-select px4_llm_control
```

Note: `test/` is currently empty — there are no `test_flake8.py` / `test_pep257.py` /
`test_copyright.py` files yet, so `colcon test` is currently a no-op for this package.

## Architecture

### Data flow

```
command_gui / command_cli  --/nl_command (String)-->  mission_executor
                            <--/nl_mission/status (String)--
```

`mission_executor.py` is the core node (`nl_mission_executor`). On each `/nl_command`
message it:
1. Snapshots the drone's current `x, y, z, heading` (NED).
2. Pushes `(instruction, snapshot)` onto a queue for a background **planner thread**
   (`_planner_worker`). The Anthropic API call blocks on the network and must never stall
   the 10 Hz offboard heartbeat, so it never runs on the main/tick thread.
3. The worker calls `LLMPlanner.plan()` (`llm_planner.py`), which forces Claude to call a
   `submit_mission_plan` tool and returns a validated list of step dicts
   (`action` ∈ `takeoff | goto | move | hold | velocity | attitude | follow | land | rtl`,
   with NED `x/y/z`, `altitude`, `yaw`, `forward/right/dz/yaw_delta`, `duration`,
   `vx/vy/vz/forward_speed/right_speed/yawspeed`, `roll/pitch/thrust`, or `target`/
   `description` (for `follow`, a lowercase COCO class name plus an optional free-text
   phrase distinguishing which instance, e.g. "wearing a green shirt") fields as
   appropriate).
4. Results are drained back on the main thread (`_drain_plans`) and appended to a
   `deque` of pending steps.

### State machine (`mission_executor.py`)

`_tick()` runs at `TICK_HZ = 10`. It always publishes `OffboardControlMode` (with the
`position`/`velocity`/`attitude` flag set based on the current `State`) plus a matching
setpoint, then dispatches based on `State`:

- `GROUNDED` — initial state (also re-entered after landing). Sits disarmed, streaming
  heartbeat position setpoints. Once `_steps` has at least one queued step (i.e. an
  instruction has been planned) **and** `HEARTBEAT_TICKS` of heartbeat have been sent,
  it arms and engages OFFBOARD mode (mirrors the arm/offboard sequence used by the
  sibling `interceptor_mission` package), then transitions to `IDLE` to dispatch that
  step. Arming/offboard is never automatic — it only happens in response to a queued
  instruction.
- `IDLE` — holds the last commanded position and heading (`_hold_x/y/z`, `_hold_yaw`);
  pops and dispatches the next step from `_steps` if any are queued.
- `TAKEOFF` / `GOTO` — fly to `_goto_target` (x, y, z, yaw) via position setpoints;
  transition back to `IDLE` once within `POS_TOLERANCE_M` **and**, if a `yaw` was given,
  within `YAW_TOLERANCE_RAD` of it (`_at_yaw`). This makes a "turn in place" goto (same
  x/y/z, new yaw) actually wait for the rotation to finish before the next queued step
  starts, instead of completing instantly because the position is already correct.
  `_hold_yaw` is updated to the reached yaw on completion. A `move` step is converted
  into a `_goto_target` and dispatched into `GOTO` the same way (see below).
- `HOLD` — holds position until `_timer_until` (wall clock).
- `VELOCITY` — streams a `TrajectorySetpoint.velocity` (`_velocity_target`: vx, vy, vz,
  optional yawspeed) until `_timer_until`, then re-anchors `_hold_x/y/z`/`_hold_yaw` to
  wherever/whichever way the vehicle ended up and returns to `IDLE`. A `velocity` step's
  `forward_speed`/`right_speed` (body-frame, like `move`) are converted to NED `vx/vy`
  in `_dispatch_next_step` using the drone's real heading at dispatch time and added to
  any absolute `vx/vy` — the LLM never does this conversion itself.
- `ATTITUDE` — streams a `VehicleAttitudeSetpoint` (`_attitude_target`: roll, pitch, yaw,
  thrust, converted to a quaternion + body thrust by `euler_to_quaternion()`) until
  `_timer_until`, then re-anchors `_hold_x/y/z`/`_hold_yaw` and returns to `IDLE`. This is
  an open-loop maneuver — no position/altitude hold while active.
- `FOLLOW` — visually servos toward the `target` COCO class (e.g. `"person"`) using the
  latest `/yolo_result` detections (`_find_follow_target`, nearest-neighbor match to the
  previous tick's bbox center, since `tracker_node.py` exposes no track ID). Each tick:
  horizontal pixel offset `err_x` (bbox center vs. image center, normalized to `[-1, 1]`)
  drives `yawspeed` (`FOLLOW_KP_YAW`), and bbox-height error `err_size` (vs.
  `FOLLOW_TARGET_BBOX_HEIGHT_FRAC * CAMERA_HEIGHT`) drives `forward_speed`
  (`FOLLOW_KP_FORWARD`), both clamped (`FOLLOW_MAX_YAWSPEED_RADPS`,
  `FOLLOW_MAX_SPEED_MPS`) and converted to NED `vx/vy` via the drone's current heading —
  same `cos`/`sin` pattern as `move`/`velocity`. Altitude (`vz`) is held at 0 (no
  altitude tracking). If the step has a `description` (e.g. "wearing a green shirt"),
  `_request_target_identification` periodically (`FOLLOW_REIDENTIFY_INTERVAL_S`, and
  immediately on dispatch) draws numbered boxes around every `target`-class detection on
  the latest `/camera/color/image_raw` frame and sends it to `LLMPlanner.identify_target`
  on a background **vision worker thread** (Claude vision, non-blocking — same
  blocks-on-network/never-on-tick-thread reasoning as the planner thread). The result is
  drained on the main thread (`_drain_vision`, called every tick) and, if Claude picks a
  box, that detection's pixel center becomes the new `_follow_last_bbox` — re-seeding
  `_find_follow_target`'s nearest-neighbor lock onto that instance for subsequent ticks.
  A "none of these match" or error response leaves `_follow_last_bbox` unchanged (keeps
  tracking whatever was already locked rather than flapping on one bad frame). Without a
  `description`, `_find_follow_target` behaves as before (highest-confidence match when
  nothing is locked yet). `follow` has no duration: it streams indefinitely until either (a)
  a new instruction queues a step — `_steps` non-empty preempts `FOLLOW`, re-anchoring
  `_hold_x/y/z`/`_hold_yaw` and returning to `IDLE` so the new step dispatches normally —
  or (b) the target class isn't seen in `/yolo_result` for more than
  `FOLLOW_LOST_TIMEOUT_S`, in which case the drone hovers (zero velocity) while waiting,
  then re-anchors and returns to `IDLE`, reporting "lost" via `/nl_mission/status`. A
  bare "stop"/"stop following"/"cancel" is planned by the LLM as
  `{"action": "hold", "duration": 1}`, which is enough to trigger the
  `_steps`-non-empty preemption.
- `LAND` / `RTL` — one-shot handoff to PX4's `AUTO_LAND` / `AUTO_RTL`, then move to
  `EXTERNAL_WAIT`.
- `EXTERNAL_WAIT` — waits for PX4 to disarm after a LAND/RTL, clears any remaining
  queued steps, then returns to `GROUNDED` (disarmed, not in OFFBOARD) until the next
  instruction is queued. If PX4 hasn't reported `ARMING_STATE_DISARMED` within
  `EXTERNAL_WAIT_TIMEOUT_S`, gives up waiting and returns to `GROUNDED` anyway, so a
  stalled auto-disarm can't permanently strand later instructions in `_steps`.

`velocity`/`attitude`/`hold` step values from the planner are clamped to safe ranges in
`_dispatch_next_step` (`MAX_VELOCITY_MPS`, `MAX_YAWSPEED_RADPS`, `MAX_TILT_RAD`,
`MIN_THRUST`/`MAX_THRUST`, `MAX_TIMED_STEP_S`) regardless of what the LLM returns.

`move` steps (`forward`/`right`/`dz`/`yaw_delta`, body-frame relative to the drone's
heading) are converted to an absolute `_goto_target` in `_dispatch_next_step` using the
drone's *actual* `_pos.x/y/z/heading` at dispatch time — not anything the LLM computed —
so "forward"/"backward"/"left"/"right"/"turn left/right" are deterministic regardless of
LLM arithmetic. If a step has both a `yaw_delta` and a translation (`forward`/`right`/
`dz`), it's split: the rotation is dispatched first (pushing the translation-only
remainder back onto the front of `_steps`), so the turn completes (via `_at_yaw`) before
the translation — using the post-turn heading — begins.

Status/progress strings are published on `/nl_mission/status` via `_status_msg()` for
either front end to display, including a line whenever PX4's `arming_state`/`nav_state`
changes (`_cb_status`) — useful for diagnosing arm/offboard transitions during testing.

### Coordinate frame

All coordinates are PX4 local **NED**, matching `/fmu/out/vehicle_local_position_v1`:
`x` = North (m), `y` = East (m), `z` = Down (m) — negative `z` is above ground
(e.g. `z=-5` means 5 m up). Heading/yaw is radians clockwise from North. Heading-relative
phrases ("forward", "backward", "left", "right", "turn left/right") are emitted by the LLM
as `move` steps (`forward`/`right`/`dz`/`yaw_delta`) and converted to absolute NED by
`mission_executor.py` at dispatch time using the drone's real heading — the LLM never
does this conversion itself. Only instructions phrased in absolute compass/altitude terms
("5 m north", "climb to 10 m altitude") become `goto` steps with absolute `x/y/z`/`yaw`,
which the LLM computes directly from the snapshot of current state (no trigonometry
needed since north/east/down don't depend on heading).

### PX4 / uXRCE-DDS topics

- In: `/nl_command` (`std_msgs/String`)
- Out: `/nl_mission/status` (`std_msgs/String`)
- PX4 out (`/fmu/in/...`, BEST_EFFORT/TRANSIENT_LOCAL QoS): `offboard_control_mode`,
  `trajectory_setpoint`, `vehicle_attitude_setpoint`, `vehicle_command`
- PX4 in (`/fmu/out/...`, same QoS): `vehicle_local_position_v1`, `vehicle_status_v4`

### Vision pipeline ("follow" steps)

`vision.launch.py` (separate from the main launch file, since loading a YOLO model is
slow) bridges the `x500_mono_cam` SITL model's forward camera
(Gazebo `/camera/color/image_raw`) into ROS 2 via `ros_gz_image image_bridge`, then runs
`ultralytics_ros`'s `tracker_node.py` on it, publishing `ultralytics_ros/msg/YoloResult`
on `/yolo_result`. `mission_executor.py` subscribes to `/yolo_result` unconditionally
(`_cb_yolo`); if the vision pipeline isn't running, a `follow` step simply never finds
its target and reports "lost" after `FOLLOW_LOST_TIMEOUT_S` (graceful degradation, not a
hard dependency).

`CAMERA_WIDTH`/`CAMERA_HEIGHT` (1280×960) in `mission_executor.py` must match the
camera's `<width>`/`<height>` in
`Tools/simulation/gz/models/mono_cam/model.sdf` (used by `x500_mono_cam`) — if the
camera resolution there ever changes, update these constants too, since `_s_follow`'s
`err_x`/`err_size` normalization assumes them.

`tracker_node.py` does not populate any track-ID field on `Detection2D`, so
`_find_follow_target` re-identifies "the same" object across ticks heuristically: it
filters detections to the `target` COCO class, then picks whichever match is nearest
(by pixel distance) to the previous tick's bbox center (`_follow_last_bbox`), falling
back to the highest-confidence match when there's no previous bbox (i.e. right after
`follow` is dispatched).

### Claude-vision target selection (`follow` steps with a `description`)

When a `follow` step has a `description` (e.g. "follow the person wearing a green
shirt"), `mission_executor.py` also subscribes to `/camera/color/image_raw`
(`_cb_image`, via `cv_bridge`) and keeps the latest frame as `_latest_frame`.
`_request_target_identification` draws a numbered green box (`cv2.rectangle` +
`cv2.putText`) around every current `target`-class detection, JPEG-encodes the
annotated frame, and hands it off to a background **vision worker thread**
(`_vision_worker`) along with `description` and the list of candidate bbox centers —
this is the same blocks-on-network/never-on-tick-thread pattern as the planner
thread, since `LLMPlanner.identify_target` makes a Claude vision API call.

`identify_target` sends the annotated image plus a prompt asking which numbered box
matches `description`, and parses the reply for a 1-based index (or `None` for
"none"/unparseable). `_drain_vision` (called every tick) maps a returned index back
to that candidate's *captured* bbox center and sets `_follow_last_bbox` to it,
re-seeding `_find_follow_target`'s nearest-neighbor lock for subsequent ticks at full
10 Hz — Claude is only used for periodic re-identification
(`FOLLOW_REIDENTIFY_INTERVAL_S`, plus once immediately on dispatch), not per-tick
control.
