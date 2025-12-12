#!/usr/bin/env python3
"""
Simple letter display for Luma OLED screens
Displays letters on the screen with keyboard input
"""

from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import ImageFont
import time

# Initialize the OLED display
# Change address to 0x3C if that's your display address
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial)

# Load a font - using default PIL font, but you can specify a TTF
try:
    # Try to load a larger font for better visibility
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
except:
    # Fallback to default font
    font = ImageFont.load_default()

def display_letter(letter):
    """Display a single letter centered on the screen"""
    with canvas(device) as draw:
        # Get text size for centering
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate center position
        x = (device.width - text_width) // 2
        y = (device.height - text_height) // 2
        
        # Draw the letter
        draw.text((x, y), letter, fill="white", font=font)

def display_text(text):
    """Display text on the screen (can wrap if needed)"""
    with canvas(device) as draw:
        draw.text((10, 20), text, fill="white", font=font)

# Demo: cycle through alphabet
def demo_alphabet():
    """Display letters A-Z in sequence"""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    for letter in alphabet:
        display_letter(letter)
        time.sleep(0.5)

# Demo: display custom messages
def demo_messages():
    """Display some example messages"""
    messages = ["HELLO", "ASL", "SMART", "GLASSES"]
    while True:
        display_text("ASL")
    # for msg in messages:
    #     display_text(msg)
    #     time.sleep(1)

if __name__ == "__main__":
    try:
        print("Starting Luma OLED letter display...")
        print("Displaying alphabet...")
        demo_alphabet()
        
        print("\nDisplaying messages...")
        demo_messages()
        
        print("\nDemo complete!")
        
        # Clear screen at end
        device.clear()
        
    except KeyboardInterrupt:
        print("\nExiting...")
        device.clear()
    except Exception as e:
        print(f"Error: {e}")
        device.clear()