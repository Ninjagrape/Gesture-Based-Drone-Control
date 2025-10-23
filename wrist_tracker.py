# Monocular depth estimation from Mediapipe hand landmarks
import cv2
import mediapipe as mp
import numpy as np
import math

# ===================== INITIAL SETUP =====================

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Camera parameters
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
focal_length = frame_width  # Approximation for typical webcams
real_hand_length = 19.0  # cm, wrist to middle fingertip distance

# Tracking state
reference_position_3d = None
tracking_active = False


# ===================== FUNCTION DEFINITIONS =====================

def get_wrist_position_3d(frame):
    """Extract 3D wrist position from frame"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        wrist = hand_landmarks.landmark[0]
        middle_finger_tip = hand_landmarks.landmark[12]

        h, w, _ = frame.shape
        wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

        dx = (middle_finger_tip.x - wrist.x) * w
        dy = (middle_finger_tip.y - wrist.y) * h
        hand_length_pixels = math.sqrt(dx**2 + dy**2)

        # Depth estimation via similar triangles
        if hand_length_pixels > 0:
            z_depth = (focal_length * real_hand_length) / hand_length_pixels
        else:
            z_depth = 100.0

        # Convert 2D → 3D camera coordinates
        x_3d = ((wrist_x - w / 2) * z_depth) / focal_length
        y_3d = ((wrist_y - h / 2) * z_depth) / focal_length
        z_3d = z_depth

        return np.array([x_3d, y_3d, z_3d]), (wrist_x, wrist_y), hand_landmarks

    return None, None, None


def set_reference_position(position_3d):
    """Activate tracking and set reference position"""
    global reference_position_3d, tracking_active
    reference_position_3d = position_3d.copy()
    tracking_active = True
    print(f"Reference 3D position set: X={position_3d[0]:.1f}, Y={position_3d[1]:.1f}, Z={position_3d[2]:.1f} cm")


def calculate_movement_3d(current_position_3d):
    """Calculate 3D movement from reference"""
    if not tracking_active or reference_position_3d is None:
        return None

    displacement = current_position_3d - reference_position_3d
    dx, dy, dz = displacement
    distance_3d = np.linalg.norm(displacement)

    angle_xy = math.degrees(math.atan2(dy, dx))
    horizontal_dist = math.sqrt(dx**2 + dy**2)
    angle_z = math.degrees(math.atan2(dz, horizontal_dist)) if horizontal_dist > 0 else 0

    return {
        'dx': dx, 'dy': dy, 'dz': dz,
        'distance_3d': distance_3d,
        'angle_xy': angle_xy, 'angle_z': angle_z,
        'current': current_position_3d,
        'reference': reference_position_3d
    }


def reset_tracking():
    """Reset tracking"""
    global reference_position_3d, tracking_active
    reference_position_3d = None
    tracking_active = False
    print("Tracking reset")


# ===================== MAIN LOOP =====================

print("Controls:")
print("Press 'S' to set reference position")
print("Press 'R' to reset tracking")
print("Press 'Q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    pos_3d, pos_2d, hand_landmarks = get_wrist_position_3d(frame)

    if pos_3d is not None and pos_2d is not None:
        # Draw wrist point
        cv2.circle(frame, pos_2d, 10, (0, 255, 0), -1)

        # Draw hand skeleton
        if hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display current 3D position
        pos_text = f"Pos: X={pos_3d[0]:.1f} Y={pos_3d[1]:.1f} Z={pos_3d[2]:.1f} cm"
        cv2.putText(frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if tracking_active:
            movement = calculate_movement_3d(pos_3d)
            if movement:
                info_text = [
                    f"3D Distance: {movement['distance_3d']:.1f} cm",
                    f"Delta X: {movement['dx']:+.1f} cm (left/right)",
                    f"Delta Y: {movement['dy']:+.1f} cm (up/down)",
                    f"Delta Z: {movement['dz']:+.1f} cm (depth)",
                    f"XY Angle: {movement['angle_xy']:.0f}°",
                    f"Z Angle: {movement['angle_z']:.0f}°"
                ]
                y_offset = 70
                for text in info_text:
                    cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 30

                depth_color = (0, 255, 0)
                if movement['dz'] > 2:
                    depth_color = (0, 0, 255)
                elif movement['dz'] < -2:
                    depth_color = (255, 0, 0)
                cv2.circle(frame, pos_2d, 15, depth_color, 3)

    # Display tracking status
    status = "Tracking: ON" if tracking_active else "Tracking: OFF"
    status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
    cv2.putText(frame, status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow('3D Wrist Tracker', frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s') and pos_3d is not None:
        set_reference_position(pos_3d)
    elif key == ord('r'):
        reset_tracking()

cap.release()
cv2.destroyAllWindows()
