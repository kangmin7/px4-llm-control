# PX4 LLM Control

A ROS 2 package that lets you control a PX4 multicopter in natural language. Type a mission instruction into the GUI and LLM converts it into offboard waypoints that are executed in real time over uXRCE-DDS. Requires Anthropic (Claude) api.

## Running

**1. PX4 SITL:**
```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500
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

## Example instructions

```
take off to 5 metres
fly forward 10 metres
turn right 90 degrees and go forward 5 metres
return to home
land
```
