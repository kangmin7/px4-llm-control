#!/usr/bin/env python3
"""Converts natural-language mission instructions into structured PX4 mission steps.

Requires `pip install anthropic` and an ANTHROPIC_API_KEY in the environment.
Uses Claude's tool-use to force a schema-validated mission plan as output.
"""

import base64
import math
import os

from anthropic import Anthropic

DEFAULT_MODEL = os.environ.get('NL_MISSION_MODEL', 'claude-sonnet-4-6')

# All coordinates are PX4 local NED, matching /fmu/out/vehicle_local_position_v1:
#   x = North (m), y = East (m), z = Down (m) — negative z is above the ground
#   (e.g. z=-5 means 5 m up). Heading is radians clockwise from North.
SYSTEM_PROMPT = """You are a flight planner for a PX4 multicopter operating in NED coordinates:
  x = North (metres), y = East (metres), z = Down (metres) — negative z is above the ground
  (e.g. z=-5 means 5 m above ground). Heading is in radians, clockwise from North
  (0 = North, pi/2 = East, pi/-pi = South, -pi/2 = West).

You are given the drone's current state and a natural-language instruction. Call
submit_mission_plan with the ordered list of mission steps that implements it.

Two step types cover most instructions:

- "move": for anything relative to the drone's OWN heading — "go forward/backward/
  left/right N metres", "turn left/right N degrees", "climb/descend N metres", "strafe
  left 2 m and turn right 45 degrees", etc. Set only the fields the instruction implies;
  any field you omit defaults to 0:
    - "forward": metres along the current heading (negative = backward)
    - "right": metres to the right of the current heading (negative = left)
    - "dz": change in NED z, metres (negative = climb, positive = descend)
    - "yaw_delta": change in heading, radians (positive = turn right/clockwise,
      negative = turn left/counter-clockwise)
  The executor applies these using the drone's REAL heading and position at the moment
  the step actually runs, and if a step has both a "yaw_delta" and a "forward"/"right"/
  "dz", it rotates first and then translates using the new heading. So do NOT do any
  trigonometry, running-state tracking, or position math for "move" steps — just read
  the numbers straight off the instruction.

- "goto": for an ABSOLUTE target given in compass/altitude terms, independent of the
  drone's heading — "go 5 metres north and 3 metres east", "climb to 10 metres
  altitude", "fly to x=2, y=-1". Give absolute "x"/"y"/"z" (and optional "yaw"); a
  "goto" that doesn't mention altitude should keep the current z. Compute these with
  simple addition/subtraction from "Current state" — no trigonometry needed since
  north/east/down are fixed axes regardless of heading. Omit "yaw" when the instruction
  doesn't care about heading.

The instruction may describe several actions in sequence ("go forward 1 m then turn left
90 degrees", "turn right 90 degrees and go backward 3 metres", comma- or "then"-separated
clauses, etc.) — emit one step per action, in order. Consecutive "move" steps need no
shared state: each one's forward/right/dz/yaw_delta is resolved against whatever the
heading/position is when IT runs. Only track a running (x, y, z, heading) if a later step
is an absolute "goto" whose target depends on where earlier steps leave the drone.

Example: "turn right 90 degrees then go backward 10 metres" becomes two "move" steps:
(1) {"action": "move", "yaw_delta": 1.5708}; (2) {"action": "move", "forward": -10}.
Neither step needs x/y/z/yaw — the executor works out the actual NED motion from the
drone's heading at the time each step runs.

Two additional step types give finer control:

- "velocity": command a velocity for "duration" seconds, with an optional yawspeed
  (rad/s, same clockwise-from-North sign convention as heading). Use this whenever the
  instruction gives an explicit speed, e.g. "fly forward at 2 m/s for 5 seconds". Just
  like "move", give heading-relative speeds directly — do NOT do any trigonometry; the
  executor converts these to NED using the drone's real heading at the moment the step
  runs:
    - "forward_speed": m/s along the current heading (negative = backward)
    - "right_speed": m/s to the right of the current heading (negative = left)
    - "vz": NED down speed, m/s (negative = climbing)
  For an ABSOLUTE compass-direction speed instead ("fly north at 2 m/s"), use "vx"/"vy"
  (NED north/east m/s, independent of heading) — these add on top of any
  forward_speed/right_speed.
- "attitude": command a body tilt (roll, pitch in radians, FRD frame), an absolute target
  "yaw" (radians, same convention as heading), and a normalized "thrust" (0..1, where
  ~0.5 roughly hovers the default SITL vehicle) for "duration" seconds. This is a raw,
  open-loop maneuver with no position or altitude hold — only use it when the instruction
  explicitly asks for a tilt/bank/roll/pitch/thrust maneuver, keep roll/pitch small
  (well under 0.35 rad) and "duration" short (a few seconds) unless told otherwise, since
  the vehicle will drift in position and altitude for the whole duration.

One more step type:

- "follow": visually track and follow a detected object using the onboard camera —
  "follow the person", "follow that car", "follow the dog". Set "target" to the
  lowercase COCO class name of the object to follow (e.g. "person", "car", "dog",
  "cat", "bicycle"); default to "person" for instructions about following a
  person/them/him/her without naming another object. This step has no duration — the
  drone keeps following until a new instruction is given. A bare "stop"/"stop
  following"/"cancel" (with nothing else requested) becomes a single
  {"action": "hold", "duration": 1} step, which interrupts any in-progress "follow" and
  leaves the drone holding position.
  If the instruction further describes WHICH instance of "target" to follow when
  there could be more than one (e.g. "follow the person in the green shirt", "follow
  the red car", "follow the small dog"), set "description" to that distinguishing
  phrase verbatim (e.g. "wearing a green shirt", "the red one", "the small dog") so
  the executor can visually pick out that specific instance. Omit "description" (or
  leave it empty) for plain "follow the person"/"follow the car" with nothing
  distinguishing one instance from another.

- "face": hold position and rotate only to keep the target centered in the camera —
  "face the person", "look at the green person", "keep facing that car". Like "follow"
  but the drone does NOT move forward/backward — it only yaws in place. Set "target"
  to the lowercase COCO class name and optionally "description" to identify a specific
  instance. Has no duration; runs until a new instruction is given.
"""

PLAN_TOOL = {
    'name': 'submit_mission_plan',
    'description': 'Submit the ordered list of mission steps that implements the instruction.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'steps': {
                'type': 'array',
                'minItems': 1,
                'items': {
                    'type': 'object',
                    'properties': {
                        'action': {
                            'type': 'string',
                            'enum': ['takeoff', 'goto', 'move', 'hold', 'velocity', 'attitude', 'follow', 'face', 'land', 'rtl'],
                        },
                        'altitude': {
                            'type': 'number',
                            'description': 'takeoff: metres above ground, positive',
                        },
                        'x': {'type': 'number', 'description': 'goto: NED north, metres'},
                        'y': {'type': 'number', 'description': 'goto: NED east, metres'},
                        'z': {
                            'type': 'number',
                            'description': 'goto: NED down, metres (negative = above ground)',
                        },
                        'yaw': {
                            'type': 'number',
                            'description': (
                                'goto: optional target heading in radians. '
                                'attitude: required target heading in radians.'
                            ),
                        },
                        'forward': {
                            'type': 'number',
                            'description': 'move: metres along the current heading (negative = backward)',
                        },
                        'right': {
                            'type': 'number',
                            'description': 'move: metres to the right of the current heading (negative = left)',
                        },
                        'dz': {
                            'type': 'number',
                            'description': 'move: change in NED z, metres (negative = climb, positive = descend)',
                        },
                        'yaw_delta': {
                            'type': 'number',
                            'description': (
                                'move: change in heading, radians '
                                '(positive = turn right/clockwise, negative = turn left/counter-clockwise)'
                            ),
                        },
                        'forward_speed': {
                            'type': 'number',
                            'description': 'velocity: m/s along the current heading (negative = backward)',
                        },
                        'right_speed': {
                            'type': 'number',
                            'description': 'velocity: m/s to the right of the current heading (negative = left)',
                        },
                        'vx': {
                            'type': 'number',
                            'description': 'velocity: absolute NED north speed, m/s (independent of heading)',
                        },
                        'vy': {
                            'type': 'number',
                            'description': 'velocity: absolute NED east speed, m/s (independent of heading)',
                        },
                        'vz': {
                            'type': 'number',
                            'description': 'velocity: NED down speed, m/s (negative = climbing)',
                        },
                        'yawspeed': {
                            'type': 'number',
                            'description': 'velocity: optional yaw rate, rad/s (clockwise from North)',
                        },
                        'roll': {'type': 'number', 'description': 'attitude: body roll, radians (FRD)'},
                        'pitch': {'type': 'number', 'description': 'attitude: body pitch, radians (FRD)'},
                        'thrust': {
                            'type': 'number',
                            'description': 'attitude: normalized thrust, 0..1 (~0.5 roughly hovers)',
                        },
                        'duration': {
                            'type': 'number',
                            'description': 'hold / velocity / attitude: seconds',
                        },
                        'target': {
                            'type': 'string',
                            'description': 'follow: lowercase COCO object class to track (e.g. "person", "car", "dog")',
                        },
                        'description': {
                            'type': 'string',
                            'description': (
                                'follow: optional free-text phrase distinguishing which instance of '
                                '"target" to follow when there could be more than one '
                                '(e.g. "wearing a green shirt", "the red one"); omit if the '
                                'instruction does not single one out'
                            ),
                        },
                    },
                    'required': ['action'],
                },
            },
        },
        'required': ['steps'],
    },
}

VALID_ACTIONS = {'takeoff', 'goto', 'move', 'hold', 'velocity', 'attitude', 'follow', 'face', 'land', 'rtl'}
REQUIRED_FIELDS = {
    'takeoff': ('altitude',),
    'goto': ('x', 'y', 'z'),
    'move': (),
    'hold': ('duration',),
    'velocity': ('duration',),
    'attitude': ('roll', 'pitch', 'yaw', 'thrust', 'duration'),
    'follow': ('target',),
    'face': ('target',),
    'land': (),
    'rtl': (),
}
MOVE_FIELDS = ('forward', 'right', 'dz', 'yaw_delta')
VELOCITY_FIELDS = ('vx', 'vy', 'vz', 'forward_speed', 'right_speed')


class PlannerError(RuntimeError):
    """Raised when the LLM response can't be turned into a valid mission plan."""


class LLMPlanner:
    """Wraps the LLM call that turns one NL instruction into a list of mission steps."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self._client = Anthropic()
        self._model = model

    def plan(self, instruction: str, state: dict) -> list:
        """Return a validated list of mission-step dicts for `instruction`.

        `state` must provide the drone's current x/y/z (NED, metres) and heading
        (radians) so the model can resolve relative directions like "forward".
        """
        user_msg = (
            f"Current state: x={state['x']:.1f} m, y={state['y']:.1f} m, z={state['z']:.1f} m, "
            f"heading={state['heading']:.2f} rad ({math.degrees(state['heading']):.0f} deg)\n"
            f"Instruction: {instruction}"
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[PLAN_TOOL],
            tool_choice={'type': 'tool', 'name': 'submit_mission_plan'},
            messages=[{'role': 'user', 'content': user_msg}],
        )
        return self._parse(response)

    def identify_target(self, image_jpeg: bytes, description: str, num_candidates: int):
        """Pick which numbered candidate in `image_jpeg` matches `description`.

        `image_jpeg` is a JPEG-encoded camera frame with `num_candidates` bounding
        boxes drawn on it, numbered 1..num_candidates. Returns the chosen 1-based
        index, or None if no candidate matches `description`.
        """
        image_b64 = base64.b64encode(image_jpeg).decode('ascii')
        prompt = (
            f'This image has {num_candidates} numbered bounding box(es) drawn on it, '
            f'each around one candidate object. Which numbered box best matches this '
            f'description: "{description}"? Reply with ONLY the number, or "none" if '
            f'none of them match.'
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=8,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': image_b64}},
                    {'type': 'text', 'text': prompt},
                ],
            }],
        )
        text = ''.join(b.text for b in response.content if b.type == 'text')
        digits = ''.join(c for c in text if c.isdigit())
        if not digits:
            return None
        index = int(digits)
        return index if 1 <= index <= num_candidates else None

    @staticmethod
    def _parse(response) -> list:
        tool_use = next(
            (block for block in response.content if block.type == 'tool_use'), None)
        if tool_use is None:
            raise PlannerError(f'LLM did not call submit_mission_plan: {response.content}')

        steps = tool_use.input.get('steps')
        if not isinstance(steps, list) or not steps:
            raise PlannerError(f'LLM submitted no non-empty "steps" list: {tool_use.input}')

        for i, step in enumerate(steps):
            action = step.get('action')
            if action not in VALID_ACTIONS:
                raise PlannerError(f'Step {i} has unknown action {action!r}: {steps}')
            missing = [f for f in REQUIRED_FIELDS[action] if f not in step]
            if missing:
                raise PlannerError(f'Step {i} ({action}) is missing {missing}: {steps}')
            step.setdefault('yaw', None)
            if action == 'move':
                for f in MOVE_FIELDS:
                    step.setdefault(f, 0.0)
                if not any(step[f] for f in MOVE_FIELDS):
                    raise PlannerError(f'Step {i} (move) has no forward/right/dz/yaw_delta: {steps}')
            elif action == 'velocity':
                for f in VELOCITY_FIELDS:
                    step.setdefault(f, 0.0)
                step.setdefault('yawspeed', None)
            elif action == 'follow':
                step['description'] = (step.get('description') or '').strip() or None

        return steps
