import cv2
import mediapipe as mp
import numpy as np
import serial
import time

serial = serial.Serial()

serial.baudrate = 115200
serial.port = "COM8"
serial.open()
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

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)
        signal = ""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Extract hand landmarks for PIP angle calculation for all fingers
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))

                # List of finger landmarks indices in HandLandmark enum (0-indexed)
                finger_landmark_indices = [
                    mp_hands.HandLandmark.THUMB_IP,
                    mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    mp_hands.HandLandmark.RING_FINGER_PIP,
                    mp_hands.HandLandmark.PINKY_PIP,
                ]

                # Calculate and print PIP angles for all fingers
                for finger_index in finger_landmark_indices:
                    pip_landmark = landmarks[finger_index]
                    pip_angle = get_3D_angle(
                        landmarks[mp_hands.HandLandmark.WRIST],
                        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                        pip_landmark,
                    )
                    if pip_angle > 90:
                        signal = signal + "0"
                    else:
                        signal = signal + "1"

                    #
                    # print(f"Finger {finger_index} PIP Angle: {pip_angle} degrees")
        time.sleep(1)
        serial.write(signal.encode("utf-8"))
        print(signal)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
