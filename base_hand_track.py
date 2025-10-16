import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# Define the drawing function
def draw_landmarks_on_image(image, detection_result):
    """Draw hand landmarks on the image"""
    annotated_image = np.copy(image)
    
    # Get hand landmarks
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks as circles
            for landmark in hand_landmarks:
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections between landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]
            
            for start_idx, end_idx in connections:
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                x1 = int(start.x * annotated_image.shape[1])
                y1 = int(start.y * annotated_image.shape[0])
                x2 = int(end.x * annotated_image.shape[1])
                y2 = int(end.y * annotated_image.shape[0])
                cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return annotated_image

# STEP 1: Import necessary modules and initialize camera
cap = cv2.VideoCapture(0)


# STEP 2: Create a HandLandmarker object
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Process live video frames
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    
    if not ret:
        break
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # STEP 4: Detect hand landmarks
    detection_result = detector.detect(mp_image)
    
    # STEP 5: Draw landmarks on frame
    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
    
    # Convert back to BGR for display
    output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    # Display the frame
    cv2.imshow('Hand Pose Detection', output_frame)
    
    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# STEP 6: Release resources
cap.release()
cv2.destroyAllWindows()
detector = None