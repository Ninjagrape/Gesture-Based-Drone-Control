"""
monocular_depth_module.py - 3d Wrist position tracker using monocular depth estimation and One-Euro filter
"""

import numpy as np
import math
from lp_filt import OneEuroFilter


class Wrist3DTracker:
    """3D wrist position tracker using monocular depth estimation"""
    
    def __init__(self, frame_width=640):
        self.focal_length = frame_width
        
        # One Euro Filter for smooth tracking
        self.position_filter = OneEuroFilter(
            min_cutoff=1.0,
            beta=0.007,
            d_cutoff=1.0
        )
        
        # Reference position for relative movement
        self.reference_position = None
        self.mode_active = False
        
        # Movement thresholds (in cm)
        self.TRANSLATE_THRESHOLD = 3.0   # cm movement to trigger translation
        self.YAW_THRESHOLD = 5.0         # cm lateral movement for yaw
        self.ACCEL_THRESHOLD = 4.0       # cm depth change for acceleration
        
    def estimate_depth_from_hand(self, hand_landmarks, frame_shape):
        """Estimate depth using multiple bone segments"""
        h, w = frame_shape[:2]
        
        bone_segments = [
            (0, 5, 8.5, 1.0),   # Wrist to index MCP
            (0, 9, 9.0, 1.0),   # Wrist to middle MCP
            (0, 13, 8.8, 1.0),  # Wrist to ring MCP
            (0, 17, 7.5, 0.9),  # Wrist to pinky MCP
            (5, 6, 4.5, 1.2),   # Index proximal phalanx
            (9, 10, 4.8, 1.2),  # Middle proximal phalanx
            (13, 14, 4.5, 1.1), # Ring proximal phalanx
        ]
        
        depth_estimates = []
        weights = []
        
        for start_idx, end_idx, real_length, base_weight in bone_segments:
            start_pt = hand_landmarks[start_idx]
            end_pt = hand_landmarks[end_idx]
            
            # Calculate pixel distance
            dx = (end_pt.x - start_pt.x) * w
            dy = (end_pt.y - start_pt.y) * h
            pixel_length = math.sqrt(dx**2 + dy**2)
            
            if pixel_length > 5:
                # Depth via similar triangles
                z_depth = (self.focal_length * real_length) / pixel_length
                
                # Confidence weighting
                visibility_conf = (getattr(start_pt, 'visibility', 1.0) + 
                                 getattr(end_pt, 'visibility', 1.0)) / 2
                length_conf = min(pixel_length / 30.0, 1.0)
                confidence = base_weight * visibility_conf * length_conf
                
                depth_estimates.append(z_depth)
                weights.append(confidence)
        
        # Weighted average with outlier removal
        if len(depth_estimates) >= 3:
            depth_array = np.array(depth_estimates)
            weight_array = np.array(weights)
            
            # Remove outliers using MAD
            median_depth = np.median(depth_array)
            mad = np.median(np.abs(depth_array - median_depth))
            
            if mad > 0:
                outlier_mask = np.abs(depth_array - median_depth) < 2.5 * mad
                depth_array = depth_array[outlier_mask]
                weight_array = weight_array[outlier_mask]
            
            if len(depth_array) > 0 and np.sum(weight_array) > 0:
                final_depth = np.average(depth_array, weights=weight_array)
                return final_depth
            elif len(depth_array) > 0:
                final_depth = np.mean(depth_array)
                return final_depth
        
        # If we have some estimates but not enough for outlier removal
        elif len(depth_estimates) > 0:
            depth_array = np.array(depth_estimates)
            weight_array = np.array(weights)
            if np.sum(weight_array) > 0:
                return np.average(depth_array, weights=weight_array)
            else:
                return np.mean(depth_array)
        
        return 100.0  # Default depth
    
    def get_wrist_position_3d(self, hand_landmarks, frame_shape):
        """Get filtered 3D wrist position"""
        h, w = frame_shape[:2]
        wrist = hand_landmarks[0]  # WRIST is index 0
        
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        
        # Estimate depth
        z_depth = self.estimate_depth_from_hand(hand_landmarks, frame_shape)
        
        # Convert to 3D coordinates
        x_3d = ((wrist_x - w/2) * z_depth) / self.focal_length
        y_3d = ((wrist_y - h/2) * z_depth) / self.focal_length
        z_3d = z_depth
        
        raw_position = np.array([x_3d, y_3d, z_3d])
        
        # Apply filter
        filtered_position = self.position_filter.update(raw_position)
        
        return filtered_position
    
    def set_reference(self, position_3d):
        """Set reference position for a mode"""
        self.reference_position = position_3d.copy()
        self.mode_active = True
        print(f"[3D] Reference: X={position_3d[0]:.1f}, Y={position_3d[1]:.1f}, Z={position_3d[2]:.1f} cm")
    
    def clear_reference(self):
        """Clear reference and exit mode"""
        self.reference_position = None
        self.mode_active = False
        print("[3D] Reference cleared")
    
    def get_movement_commands(self, current_position):
        """
        Get movement commands based on 3D displacement from reference
        Returns list of active commands
        """
        if not self.mode_active or self.reference_position is None:
            return []
        
        displacement = current_position - self.reference_position
        dx, dy, dz = displacement
        
        commands = []
        
        # Translation commands (X and Y axes)
        if abs(dx) > self.TRANSLATE_THRESHOLD:
            commands.append(("MOVE RIGHT" if dx > 0 else "MOVE LEFT", abs(dx)))
        
        if abs(dy) > self.TRANSLATE_THRESHOLD:
            commands.append(("MOVE DOWN" if dy > 0 else "MOVE UP", abs(dy)))
        
        # Yaw commands (X axis with higher threshold)
        if abs(dx) > self.YAW_THRESHOLD:
            commands.append(("YAW RIGHT" if dx > 0 else "YAW LEFT", abs(dx)))
        
        # Acceleration commands (Z axis - depth)
        if abs(dz) > self.ACCEL_THRESHOLD:
            commands.append(("ACCELERATE" if dz < 0 else "DECELERATE", abs(dz)))
        
        return commands
    
    def get_displacement_info(self, current_position):
        """Get detailed displacement information"""
        if not self.mode_active or self.reference_position is None:
            return None
        
        displacement = current_position - self.reference_position
        distance_3d = np.linalg.norm(displacement)
        
        return {
            'dx': displacement[0],
            'dy': displacement[1],
            'dz': displacement[2],
            'distance': distance_3d
        }
    
    def reset_filter(self):
        """Reset the position filter"""
        self.position_filter.reset()

# ===================== MAIN/TEST SECTION =====================
if __name__ == "__main__":
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import os

    # Initialize MediaPipe
    model_path = 'hand_landmarker.task'
    if os.path.isabs(model_path):
        model_path = os.path.basename(model_path)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize tracker
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    tracker = Wrist3DTracker(frame_width=w)
    tracking_active = False

    # File will be opened when tracking starts
    output_file = None

    print("=" * 60)
    print("3D WRIST TRACKER - DISPLACEMENT MEASUREMENT")
    print("=" * 60)
    print("\nControls:")
    print("  'Space' - Set reference position and START recording")
    print("  'R' - Reset tracking and STOP recording")
    print("  'F' - Reset filter")
    print("  'Q' - Quit")
    print("=" * 60)
    print("\nPress SPACE to set reference and begin recording displacement\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        
        # Detect hand
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        pos_3d = None
        pos_2d = None

        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Get 3D position
            pos_3d = tracker.get_wrist_position_3d(hand_landmarks, (h, w))
            
            # Get 2D wrist position for drawing
            wrist = hand_landmarks[0]
            pos_2d = (int(wrist.x * w), int(wrist.y * h))

            # Draw wrist point
            cv2.circle(frame, pos_2d, 10, (0, 255, 0), -1)

            # Draw hand skeleton
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_landmark = hand_landmarks[start_idx]
                end_landmark = hand_landmarks[end_idx]
                
                start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
                end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

            # Display current 3D position
            pos_text = f"Pos: X={pos_3d[0]:.1f} Y={pos_3d[1]:.1f} Z={pos_3d[2]:.1f} cm"
            cv2.putText(frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if tracking_active and tracker.reference_position is not None:
                # Get displacement info
                displacement = tracker.get_displacement_info(pos_3d)
                
                if displacement:
                    # Save displacement to file (same values shown on screen)
                    if output_file is not None:
                        output_file.write(f"{displacement['dx']:+.6f}, {displacement['dy']:+.6f}, {displacement['dz']:+.6f}\n")
                        output_file.flush()
                    
                    info_text = [
                        f"Displacement:",
                        f"  X: {displacement['dx']:+.1f} cm",
                        f"  Y: {displacement['dy']:+.1f} cm",
                        f"  Z: {displacement['dz']:+.1f} cm",
                        f"Total: {displacement['distance']:.1f} cm"
                    ]
                    
                    y_offset = 70
                    for text in info_text:
                        cv2.putText(frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_offset += 30

                    # Color code wrist by depth displacement
                    depth_color = (0, 255, 0)  # Green = neutral
                    if displacement['dz'] > 3:
                        depth_color = (0, 0, 255)  # Red = moving away
                    elif displacement['dz'] < -3:
                        depth_color = (255, 0, 0)  # Blue = moving closer
                    cv2.circle(frame, pos_2d, 15, depth_color, 3)

        # Display tracking status
        status = "Recording: ON" if tracking_active else "Recording: OFF"
        status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
        cv2.putText(frame, status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow('3D Wrist Tracker', frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and pos_3d is not None:
            # Start tracking and open file
            tracker.set_reference(pos_3d)
            tracking_active = True
            
            # Open new file for this recording session
            if output_file is not None:
                output_file.close()
            output_file = open('filter_testing_data/monocular_tracking_data.txt', 'w')
            output_file.write("# Monocular Depth Displacement Data\n")
            output_file.write("# delta_x, delta_y, delta_z (in cm, relative to reference)\n")
            print("\Started recording displacement to: monocular_tracking_data.txt")
            
        elif key == ord('r'):
            # Stop tracking and close file
            tracker.clear_reference()
            tracking_active = False
            
            if output_file is not None:
                output_file.close()
                output_file = None
                print("\nStopped recording. Data saved to: monocular_tracking_data.txt")
                
        elif key == ord('f'):
            tracker.reset_filter()

    # Close file and cleanup
    if output_file is not None:
        output_file.close()
        print(f"\nFinal data saved to: monocular_tracking_data.txt")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()