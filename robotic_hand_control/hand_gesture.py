import mediapipe as mp
import cv2


def get_3D_angle(p1, p2, p3):
    # Calculate the vectors between the points
    v1 = p2 - p1
    v2 = p3 - p2
    # Calculate the dot product of the vectors
    dot_product = np.dot(v1, v2)
    # Calculate the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    # Calculate the angle in radians
    angle_rad = np.arccos(cos_angle)
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    return angle_deg


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while True:
    ret, image = cap.read()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mphands.HAND_CONNECTIONS)
        print(results.multi_hand_landmarks[-1])
    # Add your hand tracking and drawing code here

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotated_image = image.copy()
    if not results.multi_hand_landmarks:
        continue
    for h in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image,
            h,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    if not results.multi_hand_world_landmarks:
        continue
    for hw in results.multi_hand_world_landmarks:
        mp_drawing.plot_landmarks(hw, mp_hands.HAND_CONNECTIONS, azimuth=5)

        WRIST = hw.landmark[mp_hands.HandLandmark.WRIST]
        MIDDLE_FINGER_MCP = hw.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        MIDDLE_FINGER_PIP = hw.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        MIDDLE_FINGER_DIP = hw.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        MIDDLE_FINGER_TIP = hw.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # NUMPY
        W = np.array([WRIST.x, WRIST.y, WRIST.z])
        D3MCP = np.array(
            [MIDDLE_FINGER_MCP.x, MIDDLE_FINGER_MCP.y, MIDDLE_FINGER_MCP.z]
        )
        D3PIP = np.array(
            [MIDDLE_FINGER_PIP.x, MIDDLE_FINGER_PIP.y, MIDDLE_FINGER_PIP.z]
        )
        D3DIP = np.array(
            [MIDDLE_FINGER_DIP.x, MIDDLE_FINGER_DIP.y, MIDDLE_FINGER_DIP.z]
        )
        D3TIP = np.array(
            [MIDDLE_FINGER_TIP.x, MIDDLE_FINGER_TIP.y, MIDDLE_FINGER_TIP.z]
        )

        # JOINTS
        JD3MCP = get_3D_angle(W, D3MCP, D3PIP)
        JD3PIP = get_3D_angle(D3MCP, D3PIP, D3DIP)
        JD3DIP = get_3D_angle(D3PIP, D3DIP, D3TIP)

        print("L'angle JD3MCP:", JD3MCP, "degrés")
        print("L'angle JD3PIP:", JD3PIP, "degrés")
        print("L'angle JD3DIP:", JD3DIP, "degrés")
    cv2.imshow("handtracker", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
