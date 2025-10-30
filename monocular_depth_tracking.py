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
        self.real_hand_length = 19.0  # cm
        
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
            (0, 5, 6.5, 1.0),   # Wrist to index MCP
            (0, 9, 7.0, 1.0),   # Wrist to middle MCP
            (0, 13, 6.8, 1.0),  # Wrist to ring MCP
            (0, 17, 5.5, 0.9),  # Wrist to pinky MCP
            (5, 6, 2.5, 1.2),   # Index proximal phalanx
            (9, 10, 2.8, 1.2),  # Middle proximal phalanx
            (13, 14, 2.5, 1.1), # Ring proximal phalanx
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

# # ===================== MAIN LOOP =====================

# print("=" * 60)
# print("3D WRIST TRACKER WITH LOW-PASS FILTERING")
# print("=" * 60)
# print("\nControls:")
# print("  'S' - Set reference position")
# print("  'R' - Reset tracking")
# print("  'F' - Reset filter")
# print("  'Q' - Quit")
# print("=" * 60)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     pos_3d, pos_2d, hand_landmarks = get_wrist_position_3d(frame)

#     if pos_3d is not None and pos_2d is not None:
#         # Draw wrist point
#         cv2.circle(frame, pos_2d, 10, (0, 255, 0), -1)

#         # Draw hand skeleton
#         if hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         # Get velocity from filter
#         velocity = position_filter.get_velocity()

#         # Display current 3D position
#         pos_text = f"Pos: X={pos_3d[0]:.1f} Y={pos_3d[1]:.1f} Z={pos_3d[2]:.1f} cm"
#         cv2.putText(frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
#         # Display velocity
#         vel_text = f"Velocity: {velocity:.1f} cm/s"
#         cv2.putText(frame, vel_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

#         if tracking_active:
#             movement = calculate_movement_3d(pos_3d)
#             if movement:
#                 info_text = [
#                     f"3D Distance: {movement['distance_3d']:.1f} cm",
#                     f"Delta X: {movement['dx']:+.1f} cm (left/right)",
#                     f"Delta Y: {movement['dy']:+.1f} cm (up/down)",
#                     f"Delta Z: {movement['dz']:+.1f} cm (depth)",
#                     f"XY Angle: {movement['angle_xy']:.0f}°",
#                     f"Z Angle: {movement['angle_z']:.0f}°"
#                 ]
#                 y_offset = 100
#                 for text in info_text:
#                     cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#                     y_offset += 30

#                 depth_color = (0, 255, 0)
#                 if movement['dz'] > 2:
#                     depth_color = (0, 0, 255)
#                 elif movement['dz'] < -2:
#                     depth_color = (255, 0, 0)
#                 cv2.circle(frame, pos_2d, 15, depth_color, 3)

#     # Display tracking status
#     status = "Tracking: ON" if tracking_active else "Tracking: OFF"
#     status_color = (0, 255, 0) if tracking_active else (0, 0, 255)
#     cv2.putText(frame, status, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

#     cv2.imshow('3D Wrist Tracker', frame)

#     # Key controls
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('s') and pos_3d is not None:
#         set_reference_position(pos_3d)
#     elif key == ord('r'):
#         reset_tracking()
#     elif key == ord('f'):
#         reset_filter()

# cap.release()
# cv2.destroyAllWindows()