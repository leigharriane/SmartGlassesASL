from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector
import cv2

# Your existing code below
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y:y+h, x:x+w]
        imgWhite = cv2.resize(imgCrop, (224, 224))  # or whatever size your model expects

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print("Pred index:", index, "prediction vector:", prediction)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
