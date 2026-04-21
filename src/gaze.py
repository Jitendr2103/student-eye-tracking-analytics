import numpy as np
from src.config import *

class GazeEstimator:
    """
    Returns normalised raw gaze (iris ratio inside the eye socket) as (x, y) in [0..1].
    X: 0 = looking far left,  1 = looking far right
    Y: 0 = looking far up,    1 = looking far down
    This is an eye-movement ratio, NOT a screen coordinate — calibration maps it.
    """

    def __init__(self):
        self._smoothed = np.array([0.5, 0.5], dtype=float)
        self._alpha    = 0.18          # EMA — lower = smoother, less jitter

    def estimate(self, landmarks, w, h):
        """Returns (gaze_x, gaze_y) or None if no face detected."""
        if landmarks is None:
            return None

        def lm(idx):
            p = landmarks.landmark[idx]
            return np.array([p.x * w, p.y * h])

        # ── Horizontal (X) ──────────────────────────────────────────
        l_iris = lm(LEFT_IRIS)
        l_out  = lm(LEFT_EYE_OUTER)
        l_inn  = lm(LEFT_EYE_INNER)
        lx = self._ratio(l_iris[0], l_out[0], l_inn[0])

        r_iris = lm(RIGHT_IRIS)
        r_inn  = lm(RIGHT_EYE_INNER)
        r_out  = lm(RIGHT_EYE_OUTER)
        rx = self._ratio(r_iris[0], r_inn[0], r_out[0])

        # ── Vertical (Y) ────────────────────────────────────────────
        # Use iris Y relative to eyelid span — robust even with head tilt
        l_top  = lm(LEFT_EYE_TOP);   l_bot = lm(LEFT_EYE_BOT)
        r_top  = lm(RIGHT_EYE_TOP);  r_bot = lm(RIGHT_EYE_BOT)
        ly = self._ratio(l_iris[1], l_top[1], l_bot[1])
        ry = self._ratio(r_iris[1], r_top[1], r_bot[1])

        raw = np.array([(lx + rx) / 2.0, (ly + ry) / 2.0])

        # EMA smoothing
        self._smoothed = self._alpha * raw + (1.0 - self._alpha) * self._smoothed
        return float(self._smoothed[0]), float(self._smoothed[1])

    @staticmethod
    def _ratio(iris_coord, edge_a, edge_b):
        """Normalise iris position between two edge landmarks → [0, 1]."""
        span = edge_b - edge_a
        if abs(span) < 1e-6:
            return 0.5
        return float(np.clip((iris_coord - edge_a) / span, 0.0, 1.0))
