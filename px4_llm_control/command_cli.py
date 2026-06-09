#!/usr/bin/env python3
"""
Interactive CLI: reads lines from stdin and publishes them as `/nl_command`
instructions, and prints `/nl_mission/status` updates as they arrive.

Run alongside `mission_executor` (the launch file starts both) and type
plain-English mission commands, e.g.:

    take off to 5 metres, fly 10 metres north, then hold for 3 seconds and land
    turn right 90 degrees and go forward 8 metres
    return to launch

Lines starting with `#` (or blank lines) are ignored.
"""
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CommandCLI(Node):

    def __init__(self):
        super().__init__('nl_command_cli')
        self._pub = self.create_publisher(String, '/nl_command', 10)
        self.create_subscription(String, '/nl_mission/status', self._on_status, 10)

    def _on_status(self, msg: String):
        print(f'[status] {msg.data}', flush=True)

    def run(self):
        print('Type a mission instruction and press enter (Ctrl-C to quit).', flush=True)
        try:
            while rclpy.ok():
                line = input('> ').strip()
                if not line or line.startswith('#'):
                    continue
                self._pub.publish(String(data=line))
        except (EOFError, KeyboardInterrupt):
            pass


def main(args=None):
    rclpy.init(args=args)
    node = CommandCLI()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        node.run()
    finally:
        rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
