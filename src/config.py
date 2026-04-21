# ──────────────────────────────────────────────
#  Constants and Configurations
# ──────────────────────────────────────────────

# MediaPipe Face Mesh landmark indices
LEFT_EYE_OUTER   = 33
LEFT_EYE_INNER   = 133
LEFT_IRIS        = 468   # requires refine_landmarks=True

RIGHT_EYE_INNER  = 362
RIGHT_EYE_OUTER  = 263
RIGHT_IRIS       = 473

# Upper/lower eyelid for vertical gaze
LEFT_EYE_TOP     = 159
LEFT_EYE_BOT     = 145
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOT    = 374

# ──────────────────────────────────────────────
#  Analytics constants
# ──────────────────────────────────────────────
WINDOW_SECONDS      = 5      # sliding window length in seconds
FPS_ESTIMATE        = 20     # approx fps for window size calculation
CONFUSION_THRESHOLD = 0.80   # transitions/second → confusion (raised to avoid jitter false-positives)

# Hysteresis: gaze must be this far OUTSIDE the screen boundary before
# it counts as off-screen.  Prevents boundary jitter from causing rapid
# on/off flipping.
OFFSCREEN_MARGIN = 0.10

# Debounce: gaze must stay off-screen for this many consecutive frames
# before a distraction/transition is registered.
OFFSCREEN_DEBOUNCE_FRAMES = 6

# 5-point calibration targets (name, norm_x, norm_y on screen)
CALIB_POINTS = [
    ("Center",        0.5,  0.5),
    ("Top-Left",      0.06, 0.06),
    ("Top-Right",     0.94, 0.06),
    ("Bottom-Left",   0.06, 0.94),
    ("Bottom-Right",  0.94, 0.94),
]

CALIB_DURATION   = 2.5    # seconds per calibration point
CALIB_SKIP_SECS  = 0.6    # skip first N seconds of each point (eye still moving)
