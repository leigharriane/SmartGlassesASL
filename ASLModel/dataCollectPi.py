# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# from picamera2 import Picamera2
# import math
# import time
# import os
# from luma.core.interface.serial import i2c
# from luma.core.render import canvas
# from luma.oled.device import ssd1306
# from PIL import ImageFont
# import time

# serial = i2c(port=1, address=0x3C)
# device = ssd1306(serial)
# picam2 = Picamera2()
# config = picam2.create_preview_configuration(
#     main={"size": (640, 480), "format": "RGB888"}
# )
# picam2.configure(config)
# cap = cv2.picam2.capture_array

# detector = HandDetector(maxHands=1)
# offset = 20
# imgSize = 300
# folder = "Data/Z"
# counter = 0

# while True:
#     # frame = picam2.capture_array()
#     # cv2.imshow("PiCamera OpenCV Feed", frame)

#     success, img = cap.read()
#     hands, img = detector.findHands(img)
#     itemsinFolder = os.listdir(folder)
#     number_of_items = len(itemsinFolder)
#     counter = number_of_items
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#         imgCropShape = imgCrop.shape
#         aspectRatio = h / w
#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == ord("s"):
#         counter += 1
#         cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
#         print(counter)
#     # if number_of_items == 20:
#     #     break

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from picamera2 import Picamera2
import math
import time
import os
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import ImageFont

# Initialize OLED display
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial)

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()  # Start the camera

# Wait for camera to warm up
time.sleep(2)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Configuration
offset = 20
imgSize = 300
folder = "Data/Z"
counter = 0

# Create folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

print("Camera started. Press 's' to save images, 'q' to quit.")

try:
    while True:
        # Capture frame from Picamera2
        img = picam2.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Detect hands
        hands, img = detector.findHands(img)
        
        # Count items in folder
        itemsinFolder = os.listdir(folder)
        number_of_items = len(itemsinFolder)
        counter = number_of_items
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Create white background
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            # Crop hand region with boundary checking
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            
            imgCrop = img[y1:y2, x1:x2]
            
            if imgCrop.size > 0:  # Check if crop is valid
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    if wCal > 0:
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    if hCal > 0:
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                
                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)
        
        # Display main image
        cv2.imshow("Image", img)
        
        # Display counter on image
        cv2.putText(img, f'Images: {counter}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("s"):
            if hands and imgCrop.size > 0:
                counter += 1
                filename = f'{folder}/Image_{time.time()}.jpg'
                cv2.imwrite(filename, imgWhite)
                print(f"Saved: {filename} (Total: {counter})")
        
        elif key == ord("q"):
            print("Quitting...")
            break
        
        # Optional: Break after certain number of images
        # if counter >= 20:
        #     print("Reached target number of images.")
        #     break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped and windows closed.")