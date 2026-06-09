#!/usr/bin/env python3
"""Converts natural-language mission instructions into structured PX4 mission steps.

Requires `pip install anthropic` and an ANTHROPIC_API_KEY in the environment.
Uses Claude's tool-use to force a schema-validated mission plan as output.
"""

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

Resolve relative phrases ("go forward 5 metres", "turn left", "climb 3 metres", "come back")
using the drone's current position and heading, and convert them into absolute NED
coordinates / headings. "Forward"/"ahead" is along the current heading; "left" is -90
degrees from it and "right" is +90 degrees. A "goto" that doesn't mention altitude should
keep the current z. Omit "yaw" when the instruction doesn't care about heading.
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
                            'enum': ['takeoff', 'goto', 'hold', 'land', 'rtl'],
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
                            'description': 'goto: optional heading in radians',
                        },
                        'duration': {'type': 'number', 'description': 'hold: seconds'},
                    },
                    'required': ['action'],
                },
            },
        },
        'required': ['steps'],
    },
}

VALID_ACTIONS = {'takeoff', 'goto', 'hold', 'land', 'rtl'}
REQUIRED_FIELDS = {
    'takeoff': ('altitude',),
    'goto': ('x', 'y', 'z'),
    'hold': ('duration',),
    'land': (),
    'rtl': (),
}


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

        return steps
