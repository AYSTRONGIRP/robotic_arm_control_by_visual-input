import cv2
import mediapipe as mp
import numpy as np
import serial
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_3D_angle(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


cap = cv2.VideoCapture(1)  # 0 for default camera, change if you have multiple cameras

signal = ""

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        signal = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))

                finger_joint_indices = [
                    mp_hands.HandLandmark.THUMB_IP,
                    mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    mp_hands.HandLandmark.RING_FINGER_PIP,
                    mp_hands.HandLandmark.PINKY_PIP,
                ]

                for finger_index in finger_joint_indices:
                    pip_landmark = landmarks[finger_index]
                    mcp_landmark = landmarks[
                        finger_index - 1
                    ]  # MCP joint index is one less than PIP joint index
                    pip_angle = get_3D_angle(
                        landmarks[mp_hands.HandLandmark.WRIST],
                        mcp_landmark,
                        pip_landmark,
                    )

                    if pip_angle < 25:
                        signal = signal + "1"
                    else:
                        signal = signal + "0"
                    print(f"Finger {finger_index} PIP Angle: {pip_angle} degrees")
        # Toggle the first character
        if signal and signal[0] == "1":
            signal = "0" + signal[1:]
        elif signal and signal[0] == "0":
            signal = "1" + signal[1:]
        cv2.imshow("Hand Tracking", frame)

        print(signal)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
