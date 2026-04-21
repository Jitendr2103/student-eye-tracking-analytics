from collections import deque
from src.config import *

class SlidingWindowAnalytics:
    """Rolling window of on/off-screen flags → attention %, distractions, confusion."""

    def __init__(self, window_sec=WINDOW_SECONDS, fps=FPS_ESTIMATE):
        self._window             = deque(maxlen=window_sec * fps)
        self._total_distractions = 0
        self._prev_on            = True

        # Debounce: track how many consecutive frames the raw signal
        # has been off-screen before committing the state flip
        self._offscreen_run      = 0
        self._confirmed_on       = True   # debounced state

    def push(self, raw_on_screen: bool):
        """
        raw_on_screen: the per-frame on/off reading.
        The debounce logic requires OFFSCREEN_DEBOUNCE_FRAMES consecutive
        off-screen frames before the state is considered truly off-screen.
        Going back on-screen is immediate (snappy recovery, slow to trigger).
        """
        if raw_on_screen:
            self._offscreen_run = 0
            on_screen = True
        else:
            self._offscreen_run += 1
            # Only flip to off-screen once the run is long enough
            on_screen = self._offscreen_run < OFFSCREEN_DEBOUNCE_FRAMES
        # Count distraction on the first confirmed off-screen frame
        if self._confirmed_on and not on_screen:
            self._total_distractions += 1
        self._prev_on = on_screen
        self._window.append(on_screen)

    @property
    def attention(self) -> float:
        if not self._window:
            return 100.0
        return 100.0 * sum(self._window) / len(self._window)

    @property
    def distractions(self) -> int:
        return self._total_distractions

    @property
    def confusion(self) -> str:
        transitions = sum(
            1 for i in range(1, len(self._window))
            if self._window[i] != self._window[i - 1]
        )
        duration = max(len(self._window) / FPS_ESTIMATE, 1)
        rate     = transitions / duration
        if rate > CONFUSION_THRESHOLD * 2:
            return "HIGH"
        elif rate > CONFUSION_THRESHOLD:
            return "MED"
        return "LOW"

    def reset_distractions(self):
        self._total_distractions = 0
        self._window.clear()
        self._prev_on = True
