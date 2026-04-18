from collections import defaultdict
from threading import Lock, Thread
import os
import select
import sys
import time

try:
    from pynput.keyboard import Key, KeyCode, Listener  # type: ignore
    _HAS_PYNPUT = True
except Exception:
    _HAS_PYNPUT = False

    class Key:
        space = "<space>"
        backspace = "<backspace>"

    class KeyCode:
        def __init__(self, char=None):
            self.char = char

        def __eq__(self, other):
            return isinstance(other, KeyCode) and self.char == other.char

        def __hash__(self):
            return hash(self.char)

        def __repr__(self):
            return f"KeyCode(char={self.char!r})"

    class Listener:
        def __enter__(self):
            self.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.stop()


class KeystrokeCounter(Listener):
    def __init__(self):
        self.key_count_map = defaultdict(lambda: 0)
        self.key_press_list = []
        self.lock = Lock()

        if _HAS_PYNPUT:
            super().__init__(on_press=self.on_press, on_release=self.on_release)
            self._terminal_thread = None
            self._stop_terminal = False
            self._stdin_fd = None
            self._stdin_old_attr = None
        else:
            self._terminal_thread = None
            self._stop_terminal = False
            self._stdin_fd = None
            self._stdin_old_attr = None
            super().__init__()

    def on_press(self, key):
        with self.lock:
            self.key_count_map[key] += 1
            self.key_press_list.append(key)

    def on_release(self, key):
        pass

    def _translate_terminal_char(self, char):
        if char == " ":
            return Key.space
        if char in ("\x7f", "\b"):
            return Key.backspace
        return KeyCode(char=char)

    def _terminal_loop(self):
        while not self._stop_terminal:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                continue
            char = os.read(self._stdin_fd, 1).decode(errors="ignore")
            if not char:
                continue
            key = self._translate_terminal_char(char)
            self.on_press(key)

    def start(self):
        if _HAS_PYNPUT:
            return super().start()

        if self._terminal_thread is not None:
            return self

        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError("Terminal keyboard fallback requires a TTY stdin.")

        self._stdin_fd = sys.stdin.fileno()
        self._stdin_old_attr = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)
        self._stop_terminal = False
        self._terminal_thread = Thread(target=self._terminal_loop, daemon=True)
        self._terminal_thread.start()
        return self

    def stop(self):
        if _HAS_PYNPUT:
            return super().stop()

        self._stop_terminal = True
        if self._terminal_thread is not None:
            self._terminal_thread.join(timeout=0.2)
            self._terminal_thread = None

        if self._stdin_fd is not None and self._stdin_old_attr is not None:
            import termios
            termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_old_attr)
            self._stdin_fd = None
            self._stdin_old_attr = None

    def clear(self):
        with self.lock:
            self.key_count_map = defaultdict(lambda: 0)
            self.key_press_list = []

    def __getitem__(self, key):
        with self.lock:
            return self.key_count_map[key]

    def get_press_events(self):
        with self.lock:
            events = list(self.key_press_list)
            self.key_press_list = []
            return events


if __name__ == '__main__':
    with KeystrokeCounter() as counter:
        try:
            while True:
                print('Space:', counter[Key.space])
                print('q:', counter[KeyCode(char='q')])
                time.sleep(1 / 60)
        except KeyboardInterrupt:
            events = counter.get_press_events()
            print(events)
