# ------------------------------------------------------------
# Project : Finger Counting Using Camera (All 10 Fingers)
# Author  : Ashish Kumar Rawat
# Purpose : Real-time finger counting using webcam
# Tech    : Python, OpenCV, MediaPipe
# Features: Detects both hands and counts 0–10 fingers live
# ------------------------------------------------------------


import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks
finger_tips = [4, 8, 12, 16, 20]

# Open camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    total_fingers = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_type in zip(result.multi_hand_landmarks,
                                              result.multi_handedness):

            landmarks = hand_landmarks.landmark
            fingers = []

            label = hand_type.classification[0].label  # Left or Right

            # Thumb
            if label == "Right":
                fingers.append(landmarks[4].x < landmarks[3].x)
            else:
                fingers.append(landmarks[4].x > landmarks[3].x)

            # Other 4 fingers
            for tip in finger_tips[1:]:
                fingers.append(landmarks[tip].y < landmarks[tip - 2].y)

            total_fingers += fingers.count(True)

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display finger count
    cv2.putText(
        img,
        f'Fingers: {total_fingers}',
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Finger Counting (Press Q to Exit)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
