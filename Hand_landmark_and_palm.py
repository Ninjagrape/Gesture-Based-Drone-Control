import cv2
import numpy as np
import mediapipe as mp
import time

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# OpenCV setup 
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Face detector for face exclusion
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Background motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=35, detectShadows=False)

# Skin color range in YCrCb
lower_skin = np.array([0, 133, 77], dtype=np.uint8)
upper_skin = np.array([255, 173, 127], dtype=np.uint8)

# Define different region colour for visualisation
PALM_COLOR = (0, 100, 255)
HAND_BOX_COLOR = (0, 255, 0)
CONTOUR_COLOR = (255, 150, 0)
FINGER_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 255)

# Main 
with mp_hands.Hands(
    # Live video
    static_image_mode=False,
    # Maximum tracking two hands
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
) as hands:

    prev_time = time.time()
    # Read camera frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Mirror the video image
        frame = cv2.flip(frame, 1)
        # Obatin frame dimensions
        h, w, _ = frame.shape
        # Convert to gratscale for masking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        # Create a blank section for face region
        face_mask = np.zeros((h, w), dtype=np.uint8)
        for (x, y, fw, fh) in faces:
            cv2.rectangle(face_mask, (x, y), (x + fw, y + fh), 255, -1)
        # Motion mask
        motion_mask = bg_subtractor.apply(frame)
        # Skin mask with YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        # Ignore face region and only keep skin and motion
        mask = cv2.bitwise_and(motion_mask, skin_mask)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(face_mask))
        # Clean up noise
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), 2)

        # MediaPipe landmark detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                # Draw 21 hand landmarks and finger connections
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=FINGER_COLOR, thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255,255,0), thickness=2)
                )
                # Convert to pixel coordinates
                pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in hand_landmarks.landmark])
                # Palm region contour by using 6 hand landmarks corrdinates as reference 
                palm_idx = [0, 1, 5, 9, 13, 17]
                palm_pts = pts[palm_idx]
                # Compute Convex Hull around the reference points
                hull = cv2.convexHull(palm_pts)
                # Draw palm contour 
                cv2.polylines(frame, [hull], isClosed=True, color=CONTOUR_COLOR, thickness=2)
                # Compute the palm bounding box around Convex Hull
                x_p, y_p, w_p, h_p = cv2.boundingRect(hull)
                # Draw the palm bounding box with the coordinates
                cv2.rectangle(frame, (x_p, y_p), (x_p + w_p, y_p + h_p), PALM_COLOR, 2)
                # Label the region
                cv2.putText(frame, "Palm", (x_p, y_p - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, PALM_COLOR, 2)

                # Full hand bounding box 
                xs, ys = pts[:,0], pts[:,1]
                x_min, x_max = np.min(xs), np.max(xs)
                y_min, y_max = np.min(ys), np.max(ys)
                # Draw full hand bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), HAND_BOX_COLOR, 2)
                cv2.putText(frame, "Hand", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, HAND_BOX_COLOR, 2)
                cv2.putText(frame, f"W:{w_p}px H:{h_p}px", (x_p, y_p + h_p + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, PALM_COLOR, 2)

                # Finger labeling and counting 
                tip_ids = [4, 8, 12, 16, 20]
                # Identify the finger base on the hand landmark 
                names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                fingers_up = 0
                for i, tid in enumerate(tip_ids):
                    # Fingertip y coordinate
                    tip_y = pts[tid][1]
                    # Base y coordinate
                    base_y = pts[tid - 2][1]
                    # Identify if finger raised
                    if tip_y < base_y:  
                        fingers_up += 1
                    #Label finger with name
                    cv2.putText(frame, names[i],
                                (pts[tid][0] - 20, pts[tid][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

                # Hand labels
                hand_label = handedness.classification[0].label
                cv2.putText(frame, f"{hand_label} Hand", (x_min, y_min - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"{fingers_up} Fingers Up", (x_min, y_max + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # FPS Counter on the screen
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Display windows 
        cv2.imshow("Hand detection", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=35, detectShadows=False)

cap.release()
cv2.destroyAllWindows()

