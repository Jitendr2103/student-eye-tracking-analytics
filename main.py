import threading
from src.tracker import EyeTracker
from src.ui import ControlWindow

def main():
    tracker = EyeTracker()

    # OpenCV runs in a background thread
    capture_thread = threading.Thread(target=tracker.run_capture, daemon=True)
    capture_thread.start()

    # Tkinter control window — starts hidden, shown post-calibration
    control = ControlWindow(
        on_start=tracker.start_tracking,
        on_stop=tracker.stop_tracking,
        on_quit=tracker.stop,
    )
    tracker.set_control_window(control)

    control.run()        # blocks on main thread (Tkinter requirement)

    tracker.stop()
    capture_thread.join(timeout=3)


if __name__ == "__main__":
    main()
