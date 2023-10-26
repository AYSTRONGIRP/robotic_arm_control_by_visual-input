import serial

serial = serial.Serial()

serial.baudrate = 9600
serial.port = "COM8"
serial.open()

import cv2
import mediapipe as mp
import numpy as np

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


cap = cv2.VideoCapture(0)  # 0 for default camera, change if you have multiple cameras

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

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Extract hand landmarks for 3D angle calculation
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z))

                WRIST = landmarks[mp_hands.HandLandmark.WRIST]
                MIDDLE_FINGER_MCP = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                MIDDLE_FINGER_PIP = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                MIDDLE_FINGER_DIP = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                MIDDLE_FINGER_TIP = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Calculate 3D angles
                JD3MCP = get_3D_angle(WRIST, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP)
                JD3PIP = get_3D_angle(
                    MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP
                )
                JD3DIP = get_3D_angle(
                    MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP
                )

                # print("L'angle JD3MCP:", JD3MCP, "degrés")
                print("L'angle JD3PIP:", JD3PIP, "degrés")
                # print("L'angle JD3DIP:", JD3DIP, "degrés")
                # Convert an 8-bit integer to char
                eight_bit_integer = 1
                char_value = chr(eight_bit_integer)
                # Display the resulting frame
                if JD3PIP > 90:
                    serial.write(b"char_value")
                else:
                    char_value = chr(2)
                    serial.write(b"char_value")

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
