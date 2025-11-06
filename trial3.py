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

from picamera2 import Picamera2

# Optional deps
try:
    import cv2
except Exception:
    cv2 = None

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

RUNNING = True
def _sigint(_s, _f):
    global RUNNING
    RUNNING = False
signal.signal(signal.SIGINT, _sigint)

# ---------- OLED ----------
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

# ---------- Camera ----------
def configure_camera(picam2: Picamera2, width: int, height: int, fps: int):
    cfg = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(cfg)


    # Try setting FPS (best-effort, some modes ignore)
    try:
        picam2.set_controls({"FrameRate": fps})
    except Exception:
        pass

    # Autofocus without libcamera enums:
    # Many cams use AfMode values: 0=Manual, 1=Auto, 2=Continuous (typical)
    try:
        cc = picam2.camera_controls  # dict of supported controls
        if "AfMode" in cc:
            picam2.set_controls({"AfMode": 2})  # try continuous AF
    except Exception:
        pass

def estimate_brightness_rgb(rgb):
    if cv2 is not None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray))
    y = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    return float(np.mean(y))

def mediapipe_hand_presence(rgb_frame):
    if _HANDS is None:
        return None
    res = _HANDS.process(rgb_frame)
    return True if getattr(res, "multi_hand_landmarks", None) else False

def main():
    ap = argparse.ArgumentParser(description="rpicam + OLED (no libcamera import)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=0x3C)
    ap.add_argument("--rotate", type=int, default=0, choices=[0,1,2,3])
    ap.add_argument("--preview", action="store_true", help="Show preview window if OpenCV and GUI available")
    args = ap.parse_args()

    # OLED
    try:
        oled = init_oled(args.addr, args.rotate)
    except Exception as e:
        print(f"[WARN] OLED init failed: {e}", file=sys.stderr)
        oled = None

    # Camera
    picam2 = Picamera2()
    configure_camera(picam2, args.width, args.height, args.fps)
    picam2.start()
    time.sleep(0.2)  # warm-up

    frame_times = []
    last_oled = 0.0
    oled_interval = 0.25
    hand_status = "N/A" if _HANDS is None else "NO"

    try:
        while RUNNING:
            t0 = time.time()
            rgb = picam2.capture_array()  # RGB888
            if rgb is None or rgb.size == 0:
                continue

            dt = time.time() - t0
            frame_times.append(max(dt, 1e-6))
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_val = 1.0 / mean(frame_times)

            bright = estimate_brightness_rgb(rgb)

            # Optional: sample MediaPipe occasionally
            if _HANDS is not None and int(time.time() * 10) % 3 == 0:
                try:
                    hp = mediapipe_hand_presence(rgb)
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

            # Optional preview
            if args.preview and cv2 is not None:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("rpicam", bgr)
                if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                    break

    finally:
        picam2.stop()
        if _HANDS is not None:
            _HANDS.close()
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
