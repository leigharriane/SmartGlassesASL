from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from PIL import Image, ImageDraw, ImageFont
import time

# I2C init
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial, width=128, height=64)

# Load font
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except Exception:
    font = ImageFont.load_default()

text = "Hello World!"

while True:
    device.clear()

    img = Image.new("1", (device.width, device.height))
    draw = ImageDraw.Draw(img)

    # Measure text with textbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Center text
    x = (device.width - tw) // 2
    y = (device.height - th) // 2

    draw.text((x, y), text, font=font, fill=255)

    device.display(img)

    print("? Updated display")
    time.sleep(1)  # update every second (can be removed)

