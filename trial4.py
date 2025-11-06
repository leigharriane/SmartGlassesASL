import time
import signal
import sys

# --- Camera (Picamera2 / libcamera) ---
from picamera2 import Picamera2
try:
    from libcamera import controls  # optional
except Exception:
    controls = None

# --- OLED (luma.oled) ---
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306

# --- Image/Display helpers ---
from PIL import Image, ImageDraw, ImageFont

# --- Desktop preview ---
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

RUNNING = True

def handle_sigint(signum, frame):
    global RUNNING
    RUNNING = False

signal.signal(signal.SIGINT, handle_sigint)

def text(draw, xy, msg, font=None):
    """
    Safe text draw using Pillow >=10 (textbbox instead of deprecated textsize).
    """
    draw.text(xy, msg, font=font, fill=255)

def main():
    # ---- OLED init ----
    # If your display uses a different address, change address=0x3C
    serial = i2c(port=1, address=0x3C)
    oled = ssd1306(serial, width=128, height=64)
    oled.clear()

    # Optional font (fallback to default if not found)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()


    # ---- Camera init ----
    picam2 = Picamera2()

    # A modest preview size keeps CPU low and OLED downscaling clean
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "XRGB8888"}  # good for OpenCV
    )
    picam2.configure(config)

    if controls is not None:
        # Optional: lock AE/AWB later once stable, example:
        # picam2.set_controls({"AwbEnable": True, "AeEnable": True})
        pass

    picam2.start()

    # ---- Desktop preview window ----
    if HAVE_CV2:
        cv2.namedWindow("PiCam Preview", cv2.WINDOW_AUTOSIZE)

    # ---- FPS tracking ----
    frame_count = 0
    fps = 0.0
    last_fps_t = time.time()

    # Update OLED at ~10Hz to keep it readable
    last_oled_t = 0.0
    oled_interval = 0.1

    # Initial splash on OLED
    splash = Image.new("1", (128, 64), 0)
    d = ImageDraw.Draw(splash)
    text(d, (2, 2), "Camera + OLED", font)
    text(d, (2, 18), "Starting...", font)
    oled.display(splash)

    try:
        while RUNNING:
            frame = picam2.capture_array("main")  # numpy array (H,W,4) XRGB8888

            # --- Desktop preview ---
            if HAVE_CV2:
                cv2.imshow("PiCam Preview", frame)
                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    

            # --- FPS calc ---
            frame_count += 1
            now = time.time()
            if now - last_fps_t >= 1.0:
                fps = frame_count / (now - last_fps_t)
                frame_count = 0
                last_fps_t = now

            # --- OLED update (thumbnail + text) ---
            if now - last_oled_t >= oled_interval:
                # Downscale the frame to 128x64 and convert to 1-bit for SSD1306
                # Use luminance conversion for better contrast
                img = Image.fromarray(frame)
                img = img.convert("L").resize((128, 64))
                # Simple threshold to monochrome; tweak 110?150 if needed
                img = img.point(lambda x: 255 if x > 128 else 0, mode="1")

                # Draw a tiny HUD on top
                draw = ImageDraw.Draw(img)
                hud = f"{int(fps):02d} FPS  {frame.shape[1]}x{frame.shape[0]}"
                # Put a small black strip for legibility
                draw.rectangle([(0, 0), (127, 11)], fill=0, outline=0)
                text(draw, (2, 1), hud, font)

                oled.display(img)
                last_oled_t = now

    finally:
        # Cleanup
        picam2.stop()
        if HAVE_CV2:
            cv2.destroyAllWindows()
        # Clear OLED with a goodbye message
        bye = Image.new("1", (128, 64), 0)
        d = ImageDraw.Draw(bye)
        text(d, (2, 24), "Shutting down...", font)
        oled.display(bye)
        time.sleep(0.5)
        oled.clear()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
