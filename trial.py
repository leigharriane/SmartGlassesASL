import time
import sys
import signal
from statistics import mean

import cv2
import numpy as np

from PIL import ImageFont
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

# ---------- Optional: MediaPipe hand detection ----------
USE_MEDIAPIPE = True
mp_hands = None
mp = None
hands = None
try:
    if USE_MEDIAPIPE:
        import mediapipe as mp

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
except Exception:
    # Mediapipe not available or failed to init; continue without it
    mp_hands = None
    hands = None


# ---------- Graceful exit handling ----------
RUNNING = True
def _handle_sigint(sig, frame):
    global RUNNING
    RUNNING = False
signal.signal(signal.SIGINT, _handle_sigint)


def init_oled(i2c_addr=0x3C, rotate=0):
    """Initialize SSD1306 OLED on I2C addr (default 0x3C)."""
    serial = i2c(port=1, address=i2c_addr)
    device = ssd1306(serial, rotate=rotate)
    return device


def draw_oled(device, lines):
    """Draw up to ~5 lines of text centered-left on the OLED."""
    # Use default bitmap font (fast, available)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    with canvas(device) as draw:
        # Vertical spacing
        y = 0
        for line in lines:
            # Use textbbox (Pillow >= 8.0)
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            # Left align with small padding
            x = 0
            draw.text((x, y), line, font=font, fill=255)
            y += h + 1  # 1px spacing between lines
            
def init_camera(index=0, width=640, height=480, fps=30):
    cap = cv2.VideoCapture(index)
    # Some USB cams ignore set() but we try anyway
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera at index {}.".format(index))
    # Read back actual values
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, actual_w, actual_h

def estimate_brightness(gray_frame):
    """Return a simple brightness metric 0-255."""
    return float(np.mean(gray_frame))


def detect_hand_presence_bgr(frame_bgr):
    """Return True if a hand is likely present (if MediaPipe is available)."""
    if hands is None or mp_hands is None:
        return None  # Unknown / not available

    # Convert BGR -> RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        return True
    return False


def run():
    # ---- Init OLED ----
    try:
        device = init_oled(i2c_addr=0x3C, rotate=0)
    except Exception as e:
        print(f"[WARN] OLED init failed: {e}", file=sys.stderr)
        device = None

    # ---- Init Camera ----
    cap, cw, ch = init_camera(index=0, width=640, height=480, fps=30)
    print(f"[INFO] Camera opened at {cw}x{ch}")

    # ---- Loop ----
    frame_times = []
    last_oled_update = 0
    oled_update_interval = 0.25  # seconds
    hand_status = "N/A" if hands is None else "NO"

    try:
        while RUNNING:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera read failed")
                time.sleep(0.05)
                continue

            # Compute FPS
            t1 = time.time()
            frame_time = max(1e-6, t1 - t0)
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / mean(frame_times)

            # Brightness (use grayscale mean)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bright = estimate_brightness(gray)

            # Optional MediaPipe hand detection (low rate for OLED status)
            if hands is not None:
                # Sample every 3rd frame for light CPU usage
                if int(time.time() * 10) % 3 == 0:
                    try:
                        hp = detect_hand_presence_bgr(frame)
                        if hp is True:
                            hand_status = "YES"
                        elif hp is False:
                            hand_status = "NO"
                        else:
                            hand_status = "N/A"
                    except Exception:
                        hand_status = "N/A"

            # Update OLED at a modest rate
            now = time.time()
            if device is not None and (now - last_oled_update) >= oled_update_interval:
                lines = [
                    "SmartGlassesASL",
                    f"FPS: {fps:4.1f}",
                    f"Res: {cw}x{ch}",
                    f"Bright: {bright:5.1f}",
                    f"Hand: {hand_status}",
                ]
                try:
                    draw_oled(device, lines)
                except Exception as e:
                    print(f"[WARN] OLED draw failed: {e}")
                last_oled_update = now

            # If you want a preview window and you?re running with a desktop:
            # cv2.imshow("Camera", frame)
            # if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            #     break

    finally:
        cap.release()
        # cv2.destroyAllWindows()
        if hands is not None:
            hands.close()
        if device is not None:
            # Clear the OLED on exit
            try:
                draw_oled(device, ["Shutting down...", "", "", "", ""])
            except Exception:
                pass
        print("\n[INFO] Exited cleanly.")


if __name__ == "__main__":
    run()
