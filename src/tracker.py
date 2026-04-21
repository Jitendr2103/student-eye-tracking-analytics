import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import csv
from datetime import datetime

from src.gaze import GazeEstimator
from src.calibration import Calibration, CalibrationScreen
from src.analytics import SlidingWindowAnalytics
from src.ui import ControlWindow, DashboardOverlay

def generate_report(attention, distractions, confusion, timeseries_data):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary report
    filename_summary = f"report_{ts}.csv"
    rows = [
        ["Metric",        "Value"],
        ["Timestamp",     ts],
        ["Attention (%)", f"{attention:.1f}"],
        ["Distractions",  str(distractions)],
        ["Confusion",     confusion],
    ]
    with open(filename_summary, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[Report] Saved summary → {filename_summary}")
    
    # Save timeseries report
    filename_ts = f"timeseries_{ts}.csv"
    with open(filename_ts, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TimeOffset(s)", "Attention(%)", "Distractions", "Confusion"])
        writer.writerows(timeseries_data)
    print(f"[Report] Saved timeseries → {filename_ts}")
    
    return filename_summary


class EyeTracker:
    """Orchestrates capture, detection, analytics, UI, and reporting."""

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._gaze      = GazeEstimator()
        self._calib     = Calibration()        # replaced once calibration finishes
        self._analytics = SlidingWindowAnalytics()
        self._dashboard = DashboardOverlay()

        self._tracking       = False
        self._running        = True
        self._latest_gaze    = None
        self._gaze_lock      = threading.Lock()
        self._calibrating    = True
        self._calib_screen   = None
        self._control_window = None            # set by main()

        # Session snapshot on STOP
        self._final_attention    = 0.0
        self._final_distractions = 0
        self._final_confusion    = "LOW"
        
        # Timeseries data
        self._timeseries_data = []
        self._start_time = None
        self._last_log_time = 0

    # ── public ──────────────────────────────
    def get_latest_gaze(self):
        with self._gaze_lock:
            return self._latest_gaze

    def set_control_window(self, win):
        self._control_window = win

    def start_tracking(self):
        self._analytics.reset_distractions()
        self._tracking = True
        self._start_time = time.time()
        self._timeseries_data = []
        self._last_log_time = 0

    def stop_tracking(self):
        self._tracking = False
        self._final_attention    = self._analytics.attention
        self._final_distractions = self._analytics.distractions
        self._final_confusion    = self._analytics.confusion
        generate_report(self._final_attention, self._final_distractions,
                        self._final_confusion, self._timeseries_data)

    def stop(self):
        self._running = False

    # ── capture loop (runs in background thread) ──
    def run_capture(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam.")
            return

        win = "Eye Tracking — [Q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        self._calib_screen = CalibrationScreen(
            gaze_getter=self.get_latest_gaze,
            on_done=self._on_calibration_done,
        )

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            # ── MediaPipe ──────────────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._face_mesh.process(rgb)
            lms    = (result.multi_face_landmarks[0]
                      if result.multi_face_landmarks else None)

            gaze = self._gaze.estimate(lms, fw, fh)
            with self._gaze_lock:
                self._latest_gaze = gaze

            # ── Calibration phase ───────────────────────────────────
            if self._calibrating:
                drawn = self._calib_screen.draw(frame)
                if drawn is None:
                    # CalibrationScreen returned None → calibration done
                    self._calibrating = False
                else:
                    frame = drawn

            # ── Tracking phase ──────────────────────────────────────
            else:
                if self._tracking:
                    if gaze is not None:
                        sx, sy    = self._calib.gaze_to_screen(*gaze)
                        on_screen = self._calib.is_on_screen(*gaze)
                    else:
                        sx, sy    = None, None
                        on_screen = False

                    self._analytics.push(on_screen)

                    self._dashboard.draw(
                        frame,
                        self._analytics.attention,
                        self._analytics.distractions,
                        self._analytics.confusion,
                        sx, sy,
                    )
                    
                    # Log timeseries data every 1 second
                    current_time = time.time()
                    elapsed = current_time - self._start_time
                    if elapsed - self._last_log_time >= 1.0:
                        self._timeseries_data.append([
                            int(elapsed),
                            round(self._analytics.attention, 1),
                            self._analytics.distractions,
                            self._analytics.confusion
                        ])
                        self._last_log_time = int(elapsed)

                elif not self._tracking and gaze is not None:
                    # Show gaze dot even when paused
                    sx, sy = self._calib.gaze_to_screen(*gaze)
                    gx_px  = int(np.clip(sx, 0, 1) * fw)
                    gy_px  = int(np.clip(sy, 0, 1) * fh)
                    cv2.circle(frame, (gx_px, gy_px), 9, (120, 120, 120), -1)
                    cv2.circle(frame, (gx_px, gy_px), 9, (200, 200, 200), 1)

                # Status label
                status = "TRACKING" if self._tracking else "PAUSED — press START in control window"
                cv2.putText(frame, status, (10, fh - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    # ── callback ────────────────────────────
    def _on_calibration_done(self, calib: Calibration):
        self._calib = calib
        print("[Calibration] Complete — affine map ready.")
        # Reveal the control window only now
        if self._control_window is not None:
            self._control_window.show()
