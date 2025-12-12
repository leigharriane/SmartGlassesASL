# from cvzone.HandTrackingModule import HandDetector
# import cv2
# from picamera2 import Picamera2
# import numpy as np
# import h5py
# from luma.core.interface.serial import i2c
# from luma.core.render import canvas
# from luma.oled.device import ssd1306
# from PIL import ImageFont
# import time

# serial = i2c(port=1, address=0x3C)
# device = ssd1306(serial)

# try:
#     # Try to load a larger font for better visibility
#     font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
# except:
#     # Fallback to default font
#     font = ImageFont.load_default()

# def display_letter(letter):
#     """Display a single letter centered on the screen"""
#     with canvas(device) as draw:
#         # Get text size for centering
#         bbox = draw.textbbox((0, 0), letter, font=font)
#         text_width = bbox[2] - bbox[0]
#         text_height = bbox[3] - bbox[1]
        
#         # Calculate center position
#         x = (device.width - text_width) // 2
#         y = (device.height - text_height) // 2
        
#         # Draw the letter
#         draw.text((x, y), letter, fill="white", font=font)

# # Manual model loading function for H5 files
# def load_h5_model(model_path):
#     """Load weights and structure from H5 file"""
#     import json
    
#     with h5py.File(model_path, 'r') as f:
#         # Get model config
#         if 'model_config' in f.attrs:
#             model_config = json.loads(f.attrs['model_config'])
#             print(f"Model loaded: {model_config.get('class_name', 'Unknown')}")
        
#         # For now, we'll create a simple inference function
#         # This assumes a standard CNN model structure
#         return f
    
# # Simple prediction function without TensorFlow
# class SimpleKerasModel:
#     def __init__(self, model_path):
#         self.model_path = model_path
#         # Store model info
#         with h5py.File(model_path, 'r') as f:
#             self.input_shape = (224, 224, 3)
#             if 'model_weights' in f.keys():
#                 self.num_classes = len(f['model_weights'].keys())
    
#     def predict(self, img_array):
#         # Placeholder - will use TFLite if available
#         # For now, return dummy prediction
#         return np.random.rand(1, self.num_classes)

# # Try to import TensorFlow, fall back if not available
# try:
#     from tensorflow import keras
#     model = keras.models.load_model("Model/keras_model.h5", compile=False)
#     print("Model loaded with TensorFlow/Keras")
# except Exception as e:
#     print(f"TensorFlow not available ({e}), using fallback")
#     model = SimpleKerasModel("Model/keras_model.h5")

# # Load labels
# with open("Model/labels.txt", "r") as f:
#     labels = [line.strip() for line in f.readlines()]

# # Initialize hand detector
# detector = HandDetector(maxHands=1)

# # Initialize Picamera2
# picam2 = Picamera2()

# # Configure camera for OpenCV compatibility
# config = picam2.create_preview_configuration(
#     main={"size": (640, 480), "format": "RGB888"}
# )
# picam2.configure(config)
# picam2.start()

# print("Camera initialized. Press 'q' to quit.")

# try:
#     while True:
#         # Capture frame from Pi Camera
#         img = picam2.capture_array()
        
#         # Convert from RGB to BGR for OpenCV
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#         # Detect hands
#         hands, img = detector.findHands(img)

#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand['bbox']
            
#             # Ensure coordinates are within image bounds
#             y1 = max(0, y)
#             y2 = min(img.shape[0], y + h)
#             x1 = max(0, x)
#             x2 = min(img.shape[1], x + w)
            
#             if y2 > y1 and x2 > x1:
#                 imgCrop = img[y1:y2, x1:x2]
#                 imgWhite = cv2.resize(imgCrop, (224, 224))
                
#                 # Normalize and reshape for model
#                 imgArray = np.asarray(imgWhite, dtype=np.float32).reshape(1, 224, 224, 3)
#                 imgArray = (imgArray / 127.5) - 1  # Normalize to [-1, 1]
                
#                 # Get prediction
#                 prediction = model.predict(imgArray, verbose=0)
#                 index = np.argmax(prediction)
#                 confidence = prediction[0][index]
                
#                 print(f"Pred: {labels[index]} (index: {index}, conf: {confidence:.2f})")
                
#                 # Display prediction on video
#                 cv2.putText(img, f"{labels[index]}: {confidence:.2f}", 
#                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
#                            1, (0, 255, 0), 2)
                
#                 display_letter({labels[index]})

#         cv2.imshow("Image", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# except KeyboardInterrupt:
#     print("\nStopping...")

# finally:
#     picam2.stop()
#     cv2.destroyAllWindows()
#     print("Camera released and windows closed.")

from cvzone.HandTrackingModule import HandDetector
import cv2
from picamera2 import Picamera2
import numpy as np
import h5py
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306
from PIL import ImageFont
import time

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
    with canvas(device) as draw:
        bbox = draw.textbbox((0, 0), letter, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate center position
        x = (device.width - text_width) // 2
        y = (device.height - text_height) // 2
        
        # Draw the letter
        draw.text((x, y), letter, fill="white", font=font)

# Manual model loading function for H5 files
def load_h5_model(model_path):
    """Load weights and structure from H5 file"""
    import json
    
    with h5py.File(model_path, 'r') as f:
        # Get model config
        if 'model_config' in f.attrs:
            model_config = json.loads(f.attrs['model_config'])
            print(f"Model loaded: {model_config.get('class_name', 'Unknown')}")
        
        # For now, we'll create a simple inference function
        # This assumes a standard CNN model structure
        return f
    
# Simple prediction function without TensorFlow
class SimpleKerasModel:
    def __init__(self, model_path):
        self.model_path = model_path
        # Store model info
        with h5py.File(model_path, 'r') as f:
            self.input_shape = (224, 224, 3)
            if 'model_weights' in f.keys():
                self.num_classes = len(f['model_weights'].keys())
    
    def predict(self, img_array):
        # Placeholder - will use TFLite if available
        # For now, return dummy prediction
        return np.random.rand(1, self.num_classes)

# Try to import TensorFlow, fall back if not available
try:
    from tensorflow import keras
    model = keras.models.load_model("ModelPi/keras_model.h5", compile=False)
    print("Model loaded with TensorFlow/Keras")
except Exception as e:
    print(f"TensorFlow not available ({e}), using fallback")
    model = SimpleKerasModel("ModelPi/keras_model.h5")

# Load labels
with open("ModelPi/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Initialize Picamera2
picam2 = Picamera2()

# Configure camera for OpenCV compatibility
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("Camera initialized. Press 'q' to quit.")

try:
    while True:
        # Capture frame from Pi Camera
        img = picam2.capture_array()
        
        # Convert from RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Detect hands
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Ensure coordinates are within image bounds
            y1 = max(0, y)
            y2 = min(img.shape[0], y + h)
            x1 = max(0, x)
            x2 = min(img.shape[1], x + w)
            
            if y2 > y1 and x2 > x1:
                imgCrop = img[y1:y2, x1:x2]
                imgWhite = cv2.resize(imgCrop, (224, 224))
                
                # Normalize and reshape for model
                imgArray = np.asarray(imgWhite, dtype=np.float32).reshape(1, 224, 224, 3)
                imgArray = (imgArray / 127.5) - 1  # Normalize to [-1, 1]
                
                # Get prediction
                prediction = model.predict(imgArray, verbose=0)
                index = np.argmax(prediction)
                confidence = prediction[0][index]
                
                print(f"Pred: {labels[index]} (index: {index}, conf: {confidence:.2f})")
                displetter = labels[index].split(' ', 1)[1]
                print(displetter)
                display_letter(displetter)
                # Display prediction on video
                cv2.putText(img, f"{labels[index]}: {confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

