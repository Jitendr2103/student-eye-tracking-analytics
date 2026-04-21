import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np

class ControlWindow:
    """
    Small always-on-top Tkinter panel.
    Starts HIDDEN — call show() once calibration is complete so it never
    overlaps the calibration dots.
    """

    def __init__(self, on_start, on_stop, on_quit):
        self.root = tk.Tk()
        self.root.title("Eye Tracker")
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)
        self.root.geometry("210x170+10+10")

        # Hide immediately — shown only after calibration
        self.root.withdraw()

        self._on_start = on_start
        self._on_stop  = on_stop
        self._on_quit  = on_quit

        ttk.Label(self.root, text="Eye Tracking Analytics",
                  font=("Helvetica", 11, "bold")).pack(pady=(12, 4))

        self.status_var = tk.StringVar(value="● Calibration done")
        self.status_lbl = ttk.Label(self.root, textvariable=self.status_var,
                                    foreground="steelblue")
        self.status_lbl.pack()

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_btn = ttk.Button(btn_frame, text="START",
                                    command=self._start, width=9)
        self.start_btn.grid(row=0, column=0, padx=4)

        self.stop_btn = ttk.Button(btn_frame, text="STOP",
                                   command=self._stop, state="disabled", width=9)
        self.stop_btn.grid(row=0, column=1, padx=4)

        ttk.Button(self.root, text="Quit", command=self._quit).pack()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def show(self):
        """Reveal the window after calibration."""
        self.root.after(0, self.root.deiconify)

    def _start(self):
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("● Tracking")
        self.status_lbl.config(foreground="green")
        self._on_start()

    def _stop(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("● Stopped — report saved")
        self.status_lbl.config(foreground="orange")
        self._on_stop()

    def _quit(self):
        self._on_quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


class DashboardOverlay:
    """Transparent HUD panel in the bottom-right corner of the video frame."""

    W, H   = 230, 106
    MARGIN = 12
    BG     = (18, 18, 18)
    CONF_COLOR = {"LOW": (80, 220, 80), "MED": (0, 190, 255), "HIGH": (40, 40, 240)}

    def draw(self, frame, attention, distractions, confusion, sx, sy):
        fh, fw = frame.shape[:2]
        x1 = fw - self.W - self.MARGIN
        y1 = fh - self.H - self.MARGIN
        x2 = fw - self.MARGIN
        y2 = fh - self.MARGIN

        # Semi-transparent background
        roi = frame[y1:y2, x1:x2]
        bg  = np.full_like(roi, self.BG)
        cv2.addWeighted(bg, 0.78, roi, 0.22, 0, roi)
        frame[y1:y2, x1:x2] = roi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        tx   = x1 + 10
        cv2.putText(frame, "ANALYTICS",
                    (tx, y1 + 18), font, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        att_col = (80, 220, 80) if attention >= 70 else \
                  (0, 190, 255) if attention >= 40 else (40, 40, 240)
        cv2.putText(frame, f"Attention:     {attention:5.1f}%",
                    (tx, y1 + 42), font, 0.48, att_col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Distractions:  {distractions:3d}",
                    (tx, y1 + 66), font, 0.48, (220, 220, 80), 1, cv2.LINE_AA)

        conf_col = self.CONF_COLOR.get(confusion, (200, 200, 200))
        cv2.putText(frame, f"Confusion:     {confusion}",
                    (tx, y1 + 90), font, 0.48, conf_col, 1, cv2.LINE_AA)

        # Gaze dot — calibrated screen position mapped onto frame
        if sx is not None and sy is not None:
            gx_px = int(np.clip(sx, 0.0, 1.0) * fw)
            gy_px = int(np.clip(sy, 0.0, 1.0) * fh)
            on    = 0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0
            color = (0, 230, 60) if on else (0, 60, 230)
            cv2.circle(frame, (gx_px, gy_px), 9, color, -1)
            cv2.circle(frame, (gx_px, gy_px), 9, (255, 255, 255), 1)

        return frame
