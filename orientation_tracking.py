"""
orientation_tracking.py - Hand Orientation Tracking in rotational axes
Tests roll and yaw tracking in isolation if run from here
Press SPACE to start/stop tracking rotation from reference.

Detects only ONE axis at a time (whichever has largest change)
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import os

from lp_filt import OneEuroFilter

# LANDMARK GLOBALS
WRIST = 0
THUMB_CMC = 1
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17

class HandOrientationTracker:
    """Track hand roll and yaw orientation for drone control"""
    
    def __init__(self, deadzone_degrees=10.0):
        self.deadzone = deadzone_degrees
        
        # Reference orientation
        self.reference_roll = None
        self.reference_yaw = None
        self.reference_pitch = None
        
        # Filters for smooth orientation
        self.roll_filter = OneEuroFilter(min_cutoff=0.5, beta=0.007, d_cutoff=1.0)
        self.yaw_filter = OneEuroFilter(min_cutoff=0.5, beta=0.007, d_cutoff=1.0)
        self.pitch_filter = OneEuroFilter(min_cutoff=0.5, beta=0.007, d_cutoff=1.0)
        
        # Command thresholds (in degrees from reference)
        self.roll_thresholds = {
            'small': 10.0,
            'medium': 20.0,
            'large': 35.0
        }
        
        self.yaw_thresholds = {
            'small': 15.0,
            'medium': 30.0,
            'large': 50.0
        }
        
        self.pitch_thresholds = {
            'small': 10.0,
            'medium': 20.0,
            'large': 35.0
        }
        
    def calculate_hand_orientation(self, hand_landmarks):
            """Calculate roll, pitch, yaw from wrist, thumb CMC, and pinky MCP.
            Uses proper rotation matrix decomposition to avoid axis coupling."""
            if hasattr(hand_landmarks[0], 'x'):
                pts = np.array([[lm.x, lm.y, getattr(lm, 'z', 0.0)] for lm in hand_landmarks])
            else:
                pts = np.array(hand_landmarks)

            wrist = pts[WRIST]
            thumb_cmc = pts[THUMB_CMC]
            pinky_mcp = pts[PINKY_MCP]

            # Define palm plane using wrist, thumb, and pinky
            v1 = thumb_cmc - wrist
            v2 = pinky_mcp - wrist
            
            # Validate vectors
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm < 1e-6 or v2_norm < 1e-6:
                # Degenerate case - return neutral orientation
                return {
                    'roll': 0.0,
                    'yaw': 0.0,
                    'pitch': 0.0,
                    'axes': {'x': np.array([1, 0, 0]), 'y': np.array([0, 1, 0]), 'z': np.array([0, 0, 1])}
                }

            # Build hand coordinate frame
            # X-axis: across palm (thumb → pinky direction)
            x_axis = pinky_mcp - thumb_cmc
            x_norm = np.linalg.norm(x_axis)
            if x_norm < 1e-6:
                x_axis = np.array([1, 0, 0])
            else:
                x_axis /= x_norm

            # Z-axis: normal to palm plane (pointing out of palm back)
            z_axis = np.cross(v1, v2)
            z_norm = np.linalg.norm(z_axis)
            if z_norm < 1e-6:
                z_axis = np.array([0, 0, 1])
            else:
                z_axis /= z_norm

            # Y-axis: perpendicular to both (pointing from wrist toward fingers)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis) + 1e-6

            # Re-orthogonalize X to ensure perfect orthogonality
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis) + 1e-6

            # Build rotation matrix [X Y Z] where each column is an axis
            # This represents rotation from world frame to hand frame
            R = np.column_stack([x_axis, y_axis, z_axis])

            # Extract Euler angles using proper decomposition
            # Physical meanings:
            # Roll: rotation around Y-axis (wrist twist left/right - palm rotates)
            # Pitch: rotation around X-axis (wrist up/down - fingers point up/down)
            # Yaw: rotation around Z-axis (arm rotation - pointing left/right)
            
            # Using XYZ Euler decomposition (intrinsic rotations)
            # R = Rz(yaw) * Ry(roll) * Rx(pitch)
            
            sy = np.sqrt(R[0, 0]**2 + R[0, 1]**2)
            
            singular = sy < 1e-6  # Gimbal lock check
            
            if not singular:
                pitch = np.degrees(np.arctan2(-R[1, 2], R[2, 2]))
                roll = np.degrees(np.arctan2(R[0, 2], sy))
                yaw = np.degrees(np.arctan2(-R[0, 1], R[0, 0]))
            else:
                # Gimbal lock case
                pitch = np.degrees(np.arctan2(R[2, 1], R[1, 1]))
                roll = np.degrees(np.arctan2(R[0, 2], sy))
                yaw = 0.0

            return {
                'roll': roll,
                'yaw': yaw,
                'pitch': -pitch,
                'axes': {'x': x_axis, 'y': y_axis, 'z': z_axis}
            }

    
    def get_filtered_orientation(self, hand_landmarks, timestamp=None):
        """Get orientation with temporal filtering applied"""
        raw_orientation = self.calculate_hand_orientation(hand_landmarks)
        
        filtered_roll = float(self.roll_filter.update(raw_orientation['roll'], timestamp))
        filtered_yaw = float(self.yaw_filter.update(raw_orientation['yaw'], timestamp))
        filtered_pitch = float(self.pitch_filter.update(raw_orientation['pitch'], timestamp))
        
        return {
            'roll': filtered_roll,
            'yaw': filtered_yaw,
            'pitch': filtered_pitch,
            'axes': raw_orientation['axes']
        }
    
    def set_reference(self, hand_landmarks, timestamp=None):
        """Set current orientation as reference (neutral position)"""
        orientation = self.get_filtered_orientation(hand_landmarks, timestamp)
        self.reference_roll = orientation['roll']
        self.reference_yaw = orientation['yaw']
        self.reference_pitch = orientation['pitch']
        
        print(f"[REFERENCE SET] Roll={self.reference_roll:.1f}°, "
              f"Yaw={self.reference_yaw:.1f}°, Pitch={self.reference_pitch:.1f}°")
    
    def clear_reference(self):
        """Clear reference orientation"""
        self.reference_roll = None
        self.reference_yaw = None
        self.reference_pitch = None
        # print("[REFERENCE CLEARED]")
    
    def get_rotation_commands(self, hand_landmarks, timestamp=None):
        """Get rotation commands based on orientation change from reference.
        Only returns ONE command at a time - the axis with largest change."""
        if self.reference_roll is None:
            return []
        
        current = self.get_filtered_orientation(hand_landmarks, timestamp)
        
        # Calculate deltas from reference
        delta_roll = current['roll'] - self.reference_roll
        delta_yaw = current['yaw'] - self.reference_yaw
        delta_pitch = current['pitch'] - self.reference_pitch
        
        # Find which axis has the largest absolute change
        abs_deltas = {
            'roll': abs(delta_roll),
            'yaw': abs(delta_yaw),
            'pitch': abs(delta_pitch)
        }
        
        # Get the dominant axis
        dominant_axis = max(abs_deltas, key=abs_deltas.get)
        dominant_magnitude = abs_deltas[dominant_axis]
        
        # Only issue command if dominant axis exceeds deadzone
        if dominant_magnitude <= self.deadzone:
            return []
        
        commands = []
        
        # Issue command for only the dominant axis
        if dominant_axis == 'roll':
            if delta_roll > 0:
                commands.append(("ROLL RIGHT", abs(delta_roll)))
            else:
                commands.append(("ROLL LEFT", abs(delta_roll)))
        
        elif dominant_axis == 'yaw':
            if delta_yaw > 0:
                commands.append(("YAW RIGHT", abs(delta_yaw)))
            else:
                commands.append(("YAW LEFT", abs(delta_yaw)))
        
        elif dominant_axis == 'pitch':
            if delta_pitch > 0:
                commands.append(("PITCH UP", abs(delta_pitch)))
            else:
                commands.append(("PITCH DOWN", abs(delta_pitch)))
        
        return commands
    
    def get_orientation_info(self, hand_landmarks, timestamp=None):
        """Get detailed orientation information for display"""
        current = self.get_filtered_orientation(hand_landmarks, timestamp)
        
        info = {
            'current': current,
            'reference': None,
            'deltas': None,
            'dominant_axis': None
        }
        
        if self.reference_roll is not None:
            info['reference'] = {
                'roll': self.reference_roll,
                'yaw': self.reference_yaw,
                'pitch': self.reference_pitch
            }
            
            deltas = {
                'roll': current['roll'] - self.reference_roll,
                'yaw': current['yaw'] - self.reference_yaw,
                'pitch': current['pitch'] - self.reference_pitch
            }
            info['deltas'] = deltas
            
            # Determine dominant axis
            abs_deltas = {k: abs(v) for k, v in deltas.items()}
            info['dominant_axis'] = max(abs_deltas, key=abs_deltas.get)
        
        return info
    
    def reset_filters(self):
        """Reset all orientation filters"""
        self.roll_filter.reset()
        self.yaw_filter.reset()
        self.pitch_filter.reset()

def log_keypoints_to_file(hand_landmarks, base_filename="filter_testing_data/hand_keypoints"):
    """Log wrist, thumb CMC, and pinky MCP to three separate .txt files in 3-column format."""
    if hasattr(hand_landmarks[0], 'x'):
        pts = np.array([[lm.x, lm.y, getattr(lm, 'z', 0.0)] for lm in hand_landmarks])
    else:
        pts = np.array(hand_landmarks)
    
    wrist = pts[WRIST]
    thumb_cmc = pts[THUMB_CMC]
    pinky_mcp = pts[PINKY_MCP]

    # Define filenames
    files = {
        "Wrist": f"{base_filename}_wrist.txt",
        "Thumb_CMC": f"{base_filename}_thumb.txt",
        "Pinky_MCP": f"{base_filename}_pinky.txt"
    }

    # Append data to each file
    for name, (coords, fname) in zip(
        ["Wrist", "Thumb_CMC", "Pinky_MCP"],
        [(wrist, files["Wrist"]), (thumb_cmc, files["Thumb_CMC"]), (pinky_mcp, files["Pinky_MCP"])]
    ):
        with open(fname, "a") as f:
            f.write(f"{coords[0]:.6f}, {coords[1]:.6f}, {coords[2]:.6f}\n")

def visualize_hand_axes(image, hand_landmarks, frame_shape, orientation_data):
    """Draw hand coordinate axes on image"""
    h, w = frame_shape[:2]
    
    # Convert landmarks if needed
    if hasattr(hand_landmarks[0], 'x'):
        pts = np.array([[lm.x, lm.y, getattr(lm, 'z', 0.0)] for lm in hand_landmarks])
    else:
        pts = np.array(hand_landmarks)
    
    wrist = pts[0]
    
    # Get axes from orientation data
    axes = orientation_data['axes']
    x_axis = axes['x'] * 0.15  # Scale for visibility
    y_axis = axes['y'] * 0.15
    z_axis = axes['z'] * 0.15
    
    # Draw from wrist
    origin = (int(wrist[0] * w), int(wrist[1] * h))
    
    # X-axis (Red) - across palm
    x_end = wrist + x_axis
    x_point = (int(x_end[0] * w), int(x_end[1] * h))
    cv2.arrowedLine(image, origin, x_point, (0, 0, 255), 3, tipLength=0.3)
    cv2.putText(image, "X", x_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Y-axis (Green) - along hand
    y_end = wrist + y_axis
    y_point = (int(y_end[0] * w), int(y_end[1] * h))
    cv2.arrowedLine(image, origin, y_point, (0, 255, 0), 3, tipLength=0.3)
    cv2.putText(image, "Y", y_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Z-axis (Blue) - palm normal
    z_end = wrist + z_axis
    z_point = (int(z_end[0] * w), int(z_end[1] * h))
    cv2.arrowedLine(image, origin, z_point, (255, 0, 0), 3, tipLength=0.3)
    cv2.putText(image, "Z", z_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image


def draw_landmarks_on_image(image, detection_result):
    """Draw hand landmarks and connections"""
    annotated_image = np.copy(image)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

            # Draw connections
            connections = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20)
            ]
            for start_idx, end_idx in connections:
                s = hand_landmarks[start_idx]
                e = hand_landmarks[end_idx]
                x1 = int(s.x * annotated_image.shape[1])
                y1 = int(s.y * annotated_image.shape[0])
                x2 = int(e.x * annotated_image.shape[1])
                y2 = int(e.y * annotated_image.shape[0])
                cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated_image


def main():
    # Setup camera
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # MediaPipe setup
    model_path = 'hand_landmarker.task'
    if os.path.isabs(model_path):
        model_path = os.path.basename(model_path)
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)
    
    # Initialize orientation tracker
    tracker = HandOrientationTracker(deadzone_degrees=10.0)
    
    # State
    tracking_active = False
    
    print("\n" + "="*70)
    print("HAND ORIENTATION TRACKING TEST - SINGLE AXIS MODE")
    print("="*70)
    print("\nCONTROLS:")
    print("  SPACE: Start/Stop tracking (sets/clears reference)")
    print("  R: Reset filters")
    print("  Q: Quit")
    print("\nINSTRUCTIONS:")
    print("  1. Show your hand to the camera (devil horns gesture)")
    print("  2. Press SPACE to set current orientation as reference")
    print("  3. Make ONE type of movement at a time:")
    print("     - ROLL: Rotate wrist left/right (palm faces different directions)")
    print("     - YAW: Rotate entire arm (point thumb vs pinky to sides)")
    print("     - PITCH: Flop wrist forward/backward")
    print("  4. Only the LARGEST movement will be detected")
    print("  5. Press SPACE again to stop tracking")
    print("\nAXES:")
    print("  Red (X): Across palm (thumb → pinky)")
    print("  Green (Y): Along hand (wrist → fingers)")
    print("  Blue (Z): Palm normal")
    print("="*70 + "\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        # Draw landmarks
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
        
        if detection_result.hand_landmarks:
            hand = detection_result.hand_landmarks[0]
            log_keypoints_to_file(hand)

            
            # Get orientation info
            orientation_info = tracker.get_orientation_info(hand)
            current_orient = orientation_info['current']
            
            # Draw coordinate axes
            annotated_frame = visualize_hand_axes(
                annotated_frame, hand, (h, w), current_orient
            )
            
            # Display current orientation
            y_pos = 30
            cv2.putText(annotated_frame, "CURRENT ORIENTATION:", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25
            cv2.putText(annotated_frame, 
                       f"Roll:  {current_orient['roll']:+7.1f} deg",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_pos += 20
            cv2.putText(annotated_frame, 
                       f"Yaw:   {current_orient['yaw']:+7.1f} deg",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_pos += 20
            cv2.putText(annotated_frame, 
                       f"Pitch: {current_orient['pitch']:+7.1f} deg",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # If tracking active, show deltas and commands
            if tracking_active and orientation_info['deltas']:
                deltas = orientation_info['deltas']
                dominant = orientation_info['dominant_axis']
                
                # Display deltas with dominant axis highlighted
                y_pos += 40
                cv2.putText(annotated_frame, "DELTA FROM REFERENCE:", (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_pos += 25
                
                # Highlight dominant axis
                roll_color = (0, 255, 0) if dominant == 'roll' else (100, 100, 100)
                yaw_color = (0, 255, 0) if dominant == 'yaw' else (100, 100, 100)
                pitch_color = (0, 255, 0) if dominant == 'pitch' else (100, 100, 100)
                
                cv2.putText(annotated_frame, 
                           f"dRoll:  {deltas['roll']:+7.1f} deg {'<- DOMINANT' if dominant == 'roll' else ''}",
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roll_color, 2 if dominant == 'roll' else 1)
                y_pos += 20
                cv2.putText(annotated_frame, 
                           f"dYaw:   {deltas['yaw']:+7.1f} deg {'<- DOMINANT' if dominant == 'yaw' else ''}",
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yaw_color, 2 if dominant == 'yaw' else 1)
                y_pos += 20
                cv2.putText(annotated_frame, 
                           f"dPitch: {deltas['pitch']:+7.1f} deg {'<- DOMINANT' if dominant == 'pitch' else ''}",
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, pitch_color, 2 if dominant == 'pitch' else 1)
                
                # Get and display commands
                commands = tracker.get_rotation_commands(hand)
                
                if commands:
                    y_pos += 40
                    cv2.putText(annotated_frame, "ACTIVE COMMAND:", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    y_pos += 30
                    
                    for cmd, magnitude in commands:
                        # Print to console
                        print(f"[CMD] {cmd:15s} ({magnitude:5.1f}°)")
                        
                        # Display on frame
                        cv2.putText(annotated_frame, f"{cmd} ({magnitude:.1f} deg)",
                                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                                   (0, 0, 255), 2)
                        y_pos += 25
                else:
                    y_pos += 40
                    cv2.putText(annotated_frame, "No command (below deadzone)", (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            # Status indicator
            status_color = (0, 255, 0) if tracking_active else (100, 100, 100)
            status_text = "TRACKING: ON (Single Axis Mode)" if tracking_active else "TRACKING: OFF (Press SPACE)"
            cv2.putText(annotated_frame, status_text, (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        else:
            # No hand detected
            cv2.putText(annotated_frame, "No hand detected", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display
        output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Orientation Tracking Test - Single Axis', output_frame)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("[INFO] Quitting...")
            break
        
        elif key == ord(' '):  # SPACE key
            if detection_result.hand_landmarks:
                hand = detection_result.hand_landmarks[0]
                
                if not tracking_active:
                    # Start tracking - set reference
                    tracker.set_reference(hand)
                    tracking_active = True
                    print("[INFO] Tracking STARTED - Reference set (Single Axis Mode)")
                else:
                    # Stop tracking - clear reference
                    tracker.clear_reference()
                    tracking_active = False
                    print("[INFO] Tracking STOPPED - Reference cleared")
            else:
                print("[WARN] No hand detected - cannot set reference")
        
        elif key == ord('r'):
            tracker.reset_filters()
            print("[INFO] Filters reset")
    
    cap.release()
    cv2.destroyAllWindows()
    detector = None
    
    print("\n[INFO] Test completed")


if __name__ == '__main__':
    main()