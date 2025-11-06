import argparse
import sys
import time
import signal
from statistics import mean

import numpy as np
from PIL import ImageFont
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

# Picamera2 preview backends (GUI/DRM fallback)
try:
    from picamera2.previews import Preview
except Exception:
    Preview = None

# Optional OpenCV (preview + faster conversions)
try:
    import cv2
except Exception:
    cv2 = None

# Optional MediaPipe (hand presence)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    _HANDS = mp_hands.Hands(static_image_mode=False,
                            max_num_hands=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
except Exception:
    mp_hands = None
    _HANDS = None

from picamera2 import Picamera2
try:
    from libcamera import controls
except Exception:
    controls = None

def _draw_text(img, lines, x=8, y=12, dy=18):
    """Draw simple HUD text onto a BGR image (OpenCV)."""
    if cv2 is None:
        return
    for line in lines:
        cv2.putText(img, line, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += dy


def _draw_mediapipe_landmarks(img_bgr, results):
    """Draw MediaPipe hand landmarks if available."""
    if cv2 is None or mp_hands is None or results is None:
        return
    if getattr(results, "multi_hand_landmarks", None):
        for hand_landmarks in results.multi_hand_landmarks:
            h, w = img_bgr.shape[:2]
            pts = [(int(lm.x * w), int(lm.y * h))
                   for lm in hand_landmarks.landmark]
            bones = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20)
            ]
            for i, j in bones:
                cv2.line(img_bgr, pts[i], pts[j], (0, 255, 0), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img_bgr, p, 3, (0, 200, 255), -1, cv2.LINE_AA)


RUNNING = True
def _sigint(_sig, _frm):
    global RUNNING
    RUNNING = False
signal.signal(signal.SIGINT, _sigint)


def init_oled(addr: int, rotate: int):
    ser = i2c(port=1, address=addr)
    return ssd1306(ser, rotate=rotate)


def draw_oled(device, lines):
    font = ImageFont.load_default()
    with canvas(device) as draw:
        y = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            h = bbox[3] - bbox[1]
            draw.text((0, y), line, font=font, fill=255)
            y += h + 1
            


def make_config(picam2: Picamera2, width: int, height: int, fps: int, lores=False):
    """Configure Picamera2 for RGB888 frames."""
    main_fmt = {"size": (width, height), "format": "RGB888"}
    cfg = picam2.create_preview_configuration(main=main_fmt)
    picam2.configure(cfg)
    # Target FPS (best-effort)
    try:
        picam2.set_controls({"FrameRate": fps})
    except Exception:
        pass
    # Continuous AF if supported (e.g., Arducam IMX708 AF)
    if controls is not None:
        try:
            picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        except Exception:
            pass
    return cfg


def estimate_brightness_rgb(rgb_frame: np.ndarray) -> float:
    """Mean luma estimate (0-255)."""
    if cv2 is not None:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray))
    # Numpy fallback luma
    y = (0.299 * rgb_frame[..., 0] +
         0.587 * rgb_frame[..., 1] +
         0.114 * rgb_frame[..., 2])
    return float(np.mean(y))


def main():
    ap = argparse.ArgumentParser(description="rpicam (Picamera2) + OLED status")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=0x3C)
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--preview", action="store_true", help="Show a live preview on the monitor")
    args = ap.parse_args()

    # OLED init (optional)
    try:
        oled = init_oled(addr=args.addr, rotate=args.rotate)
    except Exception as e:
        print(f"[WARN] OLED init failed: {e}", file=sys.stderr)
        oled = None

    # Camera init
    picam2 = Picamera2()
    make_config(picam2, args.width, args.height, args.fps)
    picam2.start()
    time.sleep(0.2)

    # ---- Start monitor preview (OpenCV first, then Picamera2 fallback) ----
    use_cv_preview = (cv2 is not None) and args.preview
    preview_started = False


    if args.preview and not use_cv_preview and Preview is not None:
        # Desktop window (fast OpenGL)
        try:
            picam2.start_preview(Preview.QTGL)
            preview_started = True
            print("[INFO] Using Picamera2 QTGL preview")
        except Exception:
            # No desktop? Direct-to-display works on console/tty
            try:
                picam2.start_preview(Preview.DRM)
                preview_started = True
                print("[INFO] Using Picamera2 DRM preview")
            except Exception:
                pass

    if args.preview and not use_cv_preview and not preview_started:
        print("[WARN] No OpenCV GUI and Picamera2 preview failed; cannot show monitor preview.", file=sys.stderr)

    # Rolling FPS stats + OLED cadence
    frame_times = []
    last_oled = 0.0
    oled_interval = 0.25
    hand_status = "N/A" if _HANDS is None else "NO"
    mp_results = None

    try:
        while RUNNING:
            t0 = time.time()
            frame_rgb = picam2.capture_array()  # RGB888
            if frame_rgb is None or frame_rgb.size == 0:
                continue

            dt = time.time() - t0
            frame_times.append(max(dt, 1e-6))
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_val = 1.0 / mean(frame_times)

            bright = estimate_brightness_rgb(frame_rgb)

            # MediaPipe hand detection (optional)
            if _HANDS is not None:
                try:
                    mp_results = _HANDS.process(frame_rgb)  # expects RGB
                    hp = getattr(mp_results, "multi_hand_landmarks", None)
                    hand_status = "YES" if hp else "NO"
                except Exception:
                    hand_status = "N/A"

            # OLED update
            now = time.time()
            if oled is not None and (now - last_oled) >= oled_interval:
                lines = [
                    "SmartGlassesASL",
                    f"FPS: {fps_val:4.1f}",
                    f"Res: {args.width}x{args.height}",
                    f"Bright: {bright:5.1f}",
                    f"Hand: {hand_status}",
                ]
                try:
                    draw_oled(oled, lines)
                except Exception as e:
                    print(f"[WARN] OLED draw failed: {e}", file=sys.stderr)
                last_oled = now

            # OpenCV preview path (with overlays)
            if use_cv_preview:
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                _draw_mediapipe_landmarks(bgr, mp_results)
                _draw_text(
                    bgr,
                    [
                        "SmartGlassesASL",
                        f"FPS: {fps_val:4.1f}",
                        f"Res: {args.width}x{args.height}",
                        f"Bright: {bright:5.1f}",
                        f"Hand: {hand_status}",
                    ],
                    x=8, y=20, dy=22
                )

                # One-time window init + handy size
                if 'cv_win_init' not in globals():
                    cv2.namedWindow("rpicam", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("rpicam", 960, 540)
                    globals()['cv_win_init'] = True
                    globals()['cv_fullscreen'] = False

                cv2.imshow("rpicam", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or q
                    break
                elif key == ord('f'):      # toggle fullscreen
                    globals()['cv_fullscreen'] = not globals()['cv_fullscreen']
                    cv2.setWindowProperty(
                        "rpicam",
                        cv2.WND_PROP_FULLSCREEN,
                        1 if globals()['cv_fullscreen'] else 0
                    )


    finally:
        picam2.stop()
        if _HANDS is not None:
            try:
                _HANDS.close()
            except Exception:
                pass
        if oled is not None:
            try:
                draw_oled(oled, ["Shutting down...", "", "", "", ""])
            except Exception:
                pass
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("\n[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
