# PX4 LLM Control

A ROS 2 package that lets you control a PX4 multicopter in natural language. Type a mission instruction into the GUI and LLM converts it into offboard waypoints that are executed in real time over uXRCE-DDS. Requires Anthropic (Claude) api.

**Demo**:
[![Demo](https://img.youtube.com/vi/uGKmal0OO5M/maxresdefault.jpg)](https://www.youtube.com/watch?v=uGKmal0OO5M)

## Running

**1. PX4 SITL:**
```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_mono_cam
```

**2. DDS bridge:**
```bash
MicroXRCEAgent udp4 -p 8888
```

**3. LLM control GUI:**
```bash
export ANTHROPIC_API_KEY=your_api_key_here
ros2 launch px4_llm_control px4_llm_control.launch.py
```

**4. Vision pipeline (for "follow" commands):**
```bash
ros2 launch px4_llm_control vision.launch.py
```
Bridges the vehicle's forward camera into ROS 2 and runs YOLO object tracking on
it (first run downloads `yolov8n.pt`, needs internet).

## Example instructions

```
take off to 5 metres
go forward 5 m/s for 5 seconds
turn right 90 degrees then go backward 10 metres
follow the person
follow the person wearing a green shirt
return to home
```

When an instruction names a specific instance ("the person wearing a green shirt",
"the red car"), the executor periodically sends the camera frame to Claude's vision
API to pick out that instance among multiple detections of the same class — see
`CLAUDE.md` for details.
