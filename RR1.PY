import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the HandDetector
detector = HandDetector(detectionCon=0.7)

# Open a webcam feed
cap = cv2.VideoCapture(0)

# Define a dictionary to map the fingers to the corresponding gestures
gestures = {
    (1, 1, 1, 1, 1): "Five",
    (0, 1, 1, 0, 0): "Two",
    (0, 1, 1, 1, 0): "Three",
    (0, 1, 1, 1, 1): "Four",
    (0, 1, 0, 0, 0): "One",
    (0, 0, 0, 0, 0): "Fist",
    (1, 1, 1, 0, 0): "Four",
    (1, 1, 0, 0, 0): "Three",
    (1, 0, 0, 0, 0): "One",
    (0, 0, 0, 1, 0): "Pink",
    (0, 0, 0, 0, 1): "Thumb",
}

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Find hands in the image

    if hands:
        # Get the first hand
        hand = hands[0]

        # Get the bounding box of the hand
        x, y, w, h = hand['bbox']

        # Draw a bounding box around the hand
        cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 2)

        # Get the center of the hand
        cx, cy = hand['center']

        # Draw a circle at the center of the hand
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        # Get the fingers up
        fingers = detector.fingersUp(hand)

        # Convert the fingers list to a tuple
        fingers_tuple = tuple(fingers)

        # Get the gesture
        gesture = gestures.get(fingers_tuple, "Unknown")

        # Print the gesture
        print(gesture)

        # Display the gesture
        cv2.putText(img, gesture, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    # Display the image with detected hands
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()