#!/usr/bin/env python3
"""
Tkinter GUI for the nl_mission_control package.

Shows a scrollable status log and a text-entry bar at the bottom.
Type a plain-English instruction and press Enter (or click Send) to
publish it on /nl_command; incoming /nl_mission/status messages are
appended to the log in real time.
"""
import threading
import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CommandGUI(Node):

    def __init__(self, root: tk.Tk):
        super().__init__('nl_command_gui')
        self._root = root
        self._pub = self.create_publisher(String, '/nl_command', 10)
        self.create_subscription(String, '/nl_mission/status', self._on_status, 10)
        self._build_ui()

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _on_status(self, msg: String):
        self._root.after(0, self._append_log, f'[status] {msg.data}')

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self._root
        root.title('NL Mission Control')
        root.geometry('720x460')
        root.configure(bg='#1e1e1e')
        root.resizable(True, True)

        # ── top bar ──
        top = tk.Frame(root, bg='#1e1e1e')
        top.pack(fill=tk.X, padx=10, pady=(10, 4))
        tk.Label(
            top, text='NL Mission Control', font=('Helvetica', 14, 'bold'),
            bg='#1e1e1e', fg='#61afef',
        ).pack(side=tk.LEFT)
        self._status_dot = tk.Label(
            top, text='●', font=('Helvetica', 14),
            bg='#1e1e1e', fg='#3e4451',
        )
        self._status_dot.pack(side=tk.RIGHT)

        # ── log area ──
        self._log = scrolledtext.ScrolledText(
            root, state='disabled', wrap=tk.WORD,
            bg='#282c34', fg='#abb2bf', font=('Courier', 11),
            insertbackground='white', relief=tk.FLAT, bd=0,
        )
        self._log.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self._log.tag_config('status', foreground='#98c379')
        self._log.tag_config('sent',   foreground='#61afef')
        self._log.tag_config('ts',     foreground='#5c6370')

        # ── input row ──
        bottom = tk.Frame(root, bg='#1e1e1e')
        bottom.pack(fill=tk.X, padx=10, pady=(4, 10))

        self._entry = tk.Entry(
            bottom, font=('Helvetica', 12),
            bg='#282c34', fg='#abb2bf', insertbackground='white',
            relief=tk.FLAT, bd=4,
        )
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=6)
        self._entry.bind('<Return>', self._send)
        self._entry.focus()

        tk.Button(
            bottom, text='Send', font=('Helvetica', 11, 'bold'),
            bg='#61afef', fg='#1e1e1e', activebackground='#528baf',
            relief=tk.FLAT, padx=16, pady=6,
            command=self._send,
        ).pack(side=tk.LEFT, padx=(8, 0))

        tk.Button(
            bottom, text='Clear', font=('Helvetica', 11),
            bg='#3e4451', fg='#abb2bf', activebackground='#4b5263',
            relief=tk.FLAT, padx=10, pady=6,
            command=self._clear_log,
        ).pack(side=tk.LEFT, padx=(6, 0))

        self._append_log('Ready — type a mission instruction and press Enter.')

    def _send(self, _event=None):
        text = self._entry.get().strip()
        if not text or text.startswith('#'):
            return
        self._entry.delete(0, tk.END)
        self._append_log(f'> {text}', tag='sent')
        self._pub.publish(String(data=text))
        self._flash_dot()

    def _append_log(self, message: str, tag: str = 'status'):
        ts = datetime.now().strftime('%H:%M:%S')
        self._log.configure(state='normal')
        self._log.insert(tk.END, f'[{ts}] ', 'ts')
        self._log.insert(tk.END, message + '\n', tag)
        self._log.configure(state='disabled')
        self._log.see(tk.END)

    def _clear_log(self):
        self._log.configure(state='normal')
        self._log.delete('1.0', tk.END)
        self._log.configure(state='disabled')

    def _flash_dot(self):
        self._status_dot.configure(fg='#e5c07b')
        self._root.after(400, lambda: self._status_dot.configure(fg='#3e4451'))


def main(args=None):
    rclpy.init(args=args)
    root = tk.Tk()
    node = CommandGUI(root)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        root.mainloop()
    finally:
        rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
