import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.7)

# Open a webcam feed
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Find hands in the image

    # Display the image with detected hands
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
