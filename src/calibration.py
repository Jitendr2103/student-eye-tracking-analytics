import numpy as np
import cv2
import time
from src.config import *

class Calibration:
    """
    Fits a linear regression:  screen_pos = A @ gaze_pos + b
    so gaze readings can be mapped directly to normalised screen coordinates.
    """

    def __init__(self):
        self.done = False
        self._M   = None   # 2×3 affine matrix [A | b]

    def finish(self, per_point_avgs):
        """
        per_point_avgs: list of (gx, gy) measured gaze for each CALIB_POINTS entry.
        Fits a least-squares affine map: [sx, sy] = M @ [gx, gy, 1]
        """
        screen_pts = np.array([[nx, ny] for _, nx, ny in CALIB_POINTS])   # (5, 2)
        gaze_pts   = np.array(per_point_avgs)                              # (5, 2)

        # Build design matrix [gx, gy, 1] for each calibration point
        ones = np.ones((len(gaze_pts), 1))
        X    = np.hstack([gaze_pts, ones])   # (5, 3)

        # Solve:  X @ M.T ≈ screen_pts  →  M has shape (2, 3)
        self._M, _, _, _ = np.linalg.lstsq(X, screen_pts, rcond=None)
        self.done = True

        print(f"[Calibration] Affine map fitted.  Coefficients:\n{self._M}")

    def gaze_to_screen(self, gx, gy):
        """Map raw gaze (0-1) → calibrated screen position (0-1)."""
        if not self.done:
            return gx, gy
        v   = np.array([gx, gy, 1.0])
        pos = v @ self._M          # shape (2,)
        sx  = float(pos[0])
        sy  = float(pos[1])
        return sx, sy

    def is_on_screen(self, gx, gy):

        """
        Returns True if the calibrated screen position is within screen bounds.
        Uses OFFSCREEN_MARGIN as a dead-zone: the gaze must venture clearly
        beyond the edge before being counted as off-screen, preventing rapid
        flipping from boundary jitter.
        """
        sx, sy = self.gaze_to_screen(gx, gy)
        m = OFFSCREEN_MARGIN
        return (-m <= sx <= 1.0 + m) and (-m <= sy <= 1.0 + m)


class CalibrationScreen:
    """
    Renders the calibration dot sequence on the webcam frame.
    Skips CALIB_SKIP_SECS at the start of each point so the eye
    has time to arrive before samples are collected.
    """

    DOT_COLOR  = (0, 230, 100)
    DOT_RADIUS = 7

    def __init__(self, gaze_getter, on_done):
        self._get_gaze   = gaze_getter
        self._on_done    = on_done
        self._point_idx  = 0
        self._t_start    = time.time()
        self._samples    = []
        self._avgs       = []
        self._finished   = False

    def draw(self, frame):
        """
        Returns the annotated frame while calibration is running.
        Returns None once calibration is complete.
        """
        if self._finished:
            return None

        h, w   = frame.shape[:2]
        name, nx, ny = CALIB_POINTS[self._point_idx]
        px, py = int(nx * w), int(ny * h)
        elapsed = time.time() - self._t_start

        # ── Dark overlay to make dot stand out ──────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # ── Instruction banner ───────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 46), (0, 0, 0), -1)
        cv2.putText(frame,
                    f"Calibration  [{self._point_idx+1}/{len(CALIB_POINTS)}]"
                    f"  —  Look at the {name} dot",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # ── Dot ─────────────────────────────────────────────────────
        cv2.circle(frame, (px, py), self.DOT_RADIUS, self.DOT_COLOR, -1)
        cv2.circle(frame, (px, py), self.DOT_RADIUS + 2, (255, 255, 255), 1)

        # ── Progress arc (only during sample window) ─────────────────
        sample_elapsed = max(0.0, elapsed - CALIB_SKIP_SECS)
        sample_window  = CALIB_DURATION - CALIB_SKIP_SECS
        if sample_elapsed > 0:
            progress = min(sample_elapsed / sample_window, 1.0)
            angle    = int(360 * progress)
            cv2.ellipse(frame, (px, py),
                        (self.DOT_RADIUS + 9, self.DOT_RADIUS + 9),
                        -90, 0, angle, (255, 220, 0), 2)

        # ── Collect gaze (only after skip window) ────────────────────
        if elapsed >= CALIB_SKIP_SECS:
            gaze = self._get_gaze()
            if gaze is not None:
                self._samples.append(gaze)

        # ── Advance to next point ────────────────────────────────────
        if elapsed >= CALIB_DURATION:
            if self._samples:
                self._avgs.append((
                    float(np.mean([s[0] for s in self._samples])),
                    float(np.mean([s[1] for s in self._samples])),
                ))
            else:
                self._avgs.append((0.5, 0.5))

            self._point_idx += 1
            self._samples    = []
            self._t_start    = time.time()

            if self._point_idx >= len(CALIB_POINTS):
                self._finished = True
                calib = Calibration()
                calib.finish(self._avgs)
                self._on_done(calib)
                return None

        return frame
