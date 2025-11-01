"""
drone_simulator.py - Simple 3D drone simulator for visualizing gesture commands

Usage:
    from drone_simulator import DroneSimulator
    
    sim = DroneSimulator()
    sim.process_command("MOVE UP", 5.0)  # command, magnitude
    sim.render()
"""

import cv2
import numpy as np
from collections import deque


class DroneSimulator:
    def __init__(self, window_size=(800, 600), grid_size=50.0):
        """
        Initialize the drone simulator.
        
        Args:
            window_size: (width, height) of the display window
            grid_size: Size of the 3D grid in cm
        """
        self.width, self.height = window_size
        self.grid_size = grid_size
        
        # Drone state (position in cm, orientation in degrees)
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.orientation = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.velocity = 2.0  # cm per command
        
        # Trail history
        self.trail = deque(maxlen=25)
        self.trail.append(self.position.copy())
        
        # Camera view parameters
        self.view_angle_h = 30.0  # horizontal rotation
        self.view_angle_v = 20.0  # vertical rotation
        self.zoom = 5.0
        
        # Last command for display
        self.last_command = None
        self.last_magnitude = None
        
    def process_command(self, command, magnitude):
        """
        Process a drone command and update state.
        Coordinate system: X=right, Y=up, Z=forward
        Translation commands are relative to drone's current heading.
        
        Args:
            command: String like "MOVE UP", "YAW LEFT", etc.
            magnitude: Float value (cm for translation, deg for rotation)
        """
        self.last_command = command
        self.last_magnitude = magnitude
        
        # Rotation commands (update orientation first)
        if command == "YAW LEFT":
            self.orientation[2] += 0.25*magnitude
        elif command == "YAW RIGHT":
            self.orientation[2] -= 0.25*magnitude
        elif command == "PITCH UP":
            self.orientation[1] -= 0.25*magnitude
        elif command == "PITCH DOWN":
            self.orientation[1] += 0.25*magnitude
        elif command == "ROLL LEFT":
            self.orientation[0] -= 0.25*magnitude
        elif command == "ROLL RIGHT":
            self.orientation[0] += 0.25*magnitude
        
        # Keep orientation in reasonable range
        self.orientation = np.mod(self.orientation + 180, 360) - 180
        
        # Translation commands (relative to current heading)
        # Define movement in drone's local frame
        local_movement = np.array([0.0, 0.0, 0.0])
        
        if command == "MOVE UP":
            local_movement[1] += magnitude  # Up in drone frame
        elif command == "MOVE DOWN":
            local_movement[1] -= magnitude  # Down in drone frame
        elif command == "MOVE LEFT":
            local_movement[0] += magnitude  # Left in drone frame
        elif command == "MOVE RIGHT":
            local_movement[0] -= magnitude  # Right in drone frame
        elif command == "ACCELERATE":
            local_movement[2] += magnitude  # Forward in drone frame
        elif command == "DECELERATE":
            local_movement[2] -= magnitude  # Backward in drone frame
        
        # Rotate the local movement to world frame using current yaw
        # (only yaw affects horizontal movement; pitch/roll don't change intended direction)
        if np.linalg.norm(local_movement) > 0:
            yaw_rad = np.radians(self.orientation[2])
            
            # Rotation matrix for yaw only (around Y-axis)
            Ry = np.array([
                [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
                [0, 1, 0],
                [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
            ])
            
            world_movement = Ry @ local_movement
            self.position += world_movement
        
        # Add to trail
        self.trail.append(self.position.copy())
    
    def reset(self):
        """Reset drone to origin."""
        self.position = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0])
        self.trail.clear()
        self.trail.append(self.position.copy())
        self.last_command = None
        
    def _project_3d_to_2d(self, point_3d):
        """
        Project 3D point to 2D screen coordinates with isometric-style view.
        
        Args:
            point_3d: (x, y, z) in cm
        Returns:
            (screen_x, screen_y)
        """
        x, y, z = point_3d
        
        # Apply view rotation
        angle_h = np.radians(self.view_angle_h)
        angle_v = np.radians(self.view_angle_v)
        
        # Rotate around vertical axis (yaw)
        x_rot = x * np.cos(angle_h) - z * np.sin(angle_h)
        z_rot = x * np.sin(angle_h) + z * np.cos(angle_h)
        
        # Rotate around horizontal axis (pitch)
        y_rot = y * np.cos(angle_v) - z_rot * np.sin(angle_v)
        z_final = y * np.sin(angle_v) + z_rot * np.cos(angle_v)
        
        # Simple orthographic projection with zoom
        screen_x = int(self.width / 2 + x_rot * self.zoom)
        screen_y = int(self.height / 2 - y_rot * self.zoom)
        
        return screen_x, screen_y
    
    def _draw_grid(self, img):
        """Draw a 3D grid floor."""
        grid_step = self.grid_size / 5  # 5 lines in each direction
        color = (40, 40, 40)
        
        # Draw horizontal lines
        for i in range(-2, 3):
            z = i * grid_step
            p1 = self._project_3d_to_2d((-self.grid_size/2, -self.grid_size/2, z))
            p2 = self._project_3d_to_2d((self.grid_size/2, -self.grid_size/2, z))
            cv2.line(img, p1, p2, color, 1)
        
        # Draw vertical lines
        for i in range(-2, 3):
            x = i * grid_step
            p1 = self._project_3d_to_2d((x, -self.grid_size/2, -self.grid_size/2))
            p2 = self._project_3d_to_2d((x, -self.grid_size/2, self.grid_size/2))
            cv2.line(img, p1, p2, color, 1)
    
    def _draw_axes(self, img, origin, scale=10.0):
        """Draw XYZ axes at given origin.
        X=right (red), Y=up (green), Z=forward (blue)"""
        # X-axis (red) - points right
        p1 = self._project_3d_to_2d(origin)
        p2 = self._project_3d_to_2d(origin + np.array([scale, 0, 0]))
        cv2.arrowedLine(img, p1, p2, (0, 0, 255), 2, tipLength=0.3)
        
        # Y-axis (green) - points up
        p2 = self._project_3d_to_2d(origin + np.array([0, scale, 0]))
        cv2.arrowedLine(img, p1, p2, (0, 255, 0), 2, tipLength=0.3)
        
        # Z-axis (blue) - points forward
        p2 = self._project_3d_to_2d(origin + np.array([0, 0, scale]))
        cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2, tipLength=0.3)
    
    def _rotate_point(self, point, roll, pitch, yaw):
        """
        Apply roll, pitch, yaw rotations to a point.
        Coordinate system: X=right, Y=up, Z=forward
        Roll: rotation around Z (forward) axis - left/right tilt
        Pitch: rotation around X (right) axis - nose up/down
        Yaw: rotation around Y (up) axis - left/right turn
        Returns rotated point as numpy array.
        """
        # Convert angles to radians
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # Rotation matrices in our coordinate system:
        # Roll (rotation around Z/forward axis - bank left/right)
        Rz = np.array([
            [np.cos(roll_rad), -np.sin(roll_rad), 0],
            [np.sin(roll_rad), np.cos(roll_rad), 0],
            [0, 0, 1]
        ])
        
        # Pitch (rotation around X/right axis - nose up/down)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])
        
        # Yaw (rotation around Y/up axis - turn left/right)
        Ry = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        
        # Combined rotation (yaw * pitch * roll)
        R = Ry @ Rx @ Rz
        
        return R @ point
    
    def _draw_drone(self, img):
        """Draw the drone as a simple quadcopter shape with full 3D orientation."""
        # Drone body size
        arm_length = 8.0
        
        # Define arm positions in local drone coordinate system
        # Drone starts facing forward (+Z), arms in XZ plane
        local_arms = [
            np.array([arm_length, 0, arm_length]),    # right-front
            np.array([-arm_length, 0, arm_length]),   # left-front
            np.array([-arm_length, 0, -arm_length]),  # left-back
            np.array([arm_length, 0, -arm_length]),   # right-back
        ]
        
        # Apply full 3D rotation to each arm
        roll, pitch, yaw = self.orientation
        rotated_arms = [self._rotate_point(arm, roll, pitch, yaw) for arm in local_arms]
        
        center = self._project_3d_to_2d(self.position)
        
         # Draw arms and motors with different colors for depth visualization
        arm_colors = [
            (100, 255, 100),  # right-front: bright green
            (255, 100, 100),  # left-front: bright red
            (100, 100, 255),  # left-back: bright blue
            (255, 255, 100),  # right-back: bright cyan/yellow
        ]
        
        for i, rotated_arm in enumerate(rotated_arms):
            end_pos = self.position + rotated_arm
            end_screen = self._project_3d_to_2d(end_pos)
            
            # Arm
            cv2.line(img, center, end_screen, (200, 200, 200), 3)
            # Motor with unique color
            cv2.circle(img, end_screen, 6, arm_colors[i], -1)
            cv2.circle(img, end_screen, 6, (255, 255, 255), 1)
        
        # Center body
        cv2.circle(img, center, 8, (50, 50, 200), -1)
        cv2.circle(img, center, 8, (255, 255, 255), 2)
        
        # Direction indicator (front) - points in +Z direction of drone
        front_local = np.array([0, 0, arm_length * 1.3])
        front_rotated = self._rotate_point(front_local, roll, pitch, yaw)
        front_pos = self.position + front_rotated
        front_screen = self._project_3d_to_2d(front_pos)
        cv2.arrowedLine(img, center, front_screen, (0, 255, 255), 2, tipLength=0.4)
    
    def _draw_trail(self, img):
        """Draw the drone's path trail."""
        if len(self.trail) < 2:
            return
        
        points = [self._project_3d_to_2d(pos) for pos in self.trail]
        
        # Draw with fading color
        for i in range(len(points) - 1):
            alpha = i / len(points)
            color = (int(100 * alpha), int(200 * alpha), int(255 * alpha))
            cv2.line(img, points[i], points[i+1], color, 2)
    
    def _draw_hud(self, img):
        """Draw heads-up display with drone state."""
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Position
        cv2.putText(img, f"Position (cm):", (10, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(img, f"  X: {self.position[0]:+7.1f}", (10, y_offset), font, 0.5, (0, 0, 255), 1)
        y_offset += 20
        cv2.putText(img, f"  Y: {self.position[1]:+7.1f}", (10, y_offset), font, 0.5, (0, 255, 0), 1)
        y_offset += 20
        cv2.putText(img, f"  Z: {self.position[2]:+7.1f}", (10, y_offset), font, 0.5, (255, 0, 0), 1)
        y_offset += 30
        
        # Orientation
        cv2.putText(img, f"Orientation (deg):", (10, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(img, f"  Roll:  {self.orientation[0]:+6.1f}", (10, y_offset), font, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(img, f"  Pitch: {self.orientation[1]:+6.1f}", (10, y_offset), font, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(img, f"  Yaw:   {self.orientation[2]:+6.1f}", (10, y_offset), font, 0.5, (200, 200, 200), 1)
        y_offset += 30
        
        # Last command
        if self.last_command:
            cv2.putText(img, f"Last Command:", (10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cmd_text = f"  {self.last_command}"
            if self.last_magnitude is not None:
                unit = "deg" if "YAW" in self.last_command or "PITCH" in self.last_command or "ROLL" in self.last_command else "cm"
                cmd_text += f" ({self.last_magnitude:.1f}{unit})"
            cv2.putText(img, cmd_text, (10, y_offset), font, 0.5, (0, 255, 255), 1)
        
        # Controls
        y_offset = self.height - 80
        cv2.putText(img, "Controls:", (10, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(img, "  Arrow Keys: Rotate view", (10, y_offset), font, 0.4, (200, 200, 200), 1)
        y_offset += 18
        cv2.putText(img, "  +/- : Zoom", (10, y_offset), font, 0.4, (200, 200, 200), 1)
        y_offset += 18
        cv2.putText(img, "  R: Reset drone  Q: Quit", (10, y_offset), font, 0.4, (200, 200, 200), 1)
    
    def render(self, window_name="Drone Simulator"):
        """
        Render the current drone state.
        
        Returns:
            True if window is still open, False if closed
        """
        # Create blank image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw components
        self._draw_grid(img)
        self._draw_axes(img, np.array([0, -self.grid_size/2, 0]), scale=10.0)
        self._draw_trail(img)
        self._draw_drone(img)
        self._draw_hud(img)
        
        # Show
        cv2.imshow(window_name, img)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.reset()
        elif key == 82:  # Up arrow
            self.view_angle_v = max(-45, self.view_angle_v - 5)
        elif key == 84:  # Down arrow
            self.view_angle_v = min(45, self.view_angle_v + 5)
        elif key == 81:  # Left arrow
            self.view_angle_h -= 5
        elif key == 83:  # Right arrow
            self.view_angle_h += 5
        elif key == ord('+') or key == ord('='):
            self.zoom = min(20, self.zoom + 0.5)
        elif key == ord('-') or key == ord('_'):
            self.zoom = max(1, self.zoom - 0.5)
        
        return True


# Example usage
if __name__ == "__main__":
    import time
    
    sim = DroneSimulator()
    
    # Test sequence
    commands = [
        ("MOVE UP", 10.0),
        ("MOVE RIGHT", 15.0),
        ("YAW RIGHT", 45.0),
        ("ACCELERATE", 20.0),
        ("MOVE LEFT", 10.0),
        ("YAW LEFT", 90.0),
        ("DECELERATE", 15.0),
    ]
    
    cmd_idx = 0
    last_cmd_time = time.time()
    
    print("Drone Simulator - Running test sequence")
    print("Press Q to quit, R to reset, Arrow keys to rotate view, +/- to zoom")
    
    while sim.render():
        # Execute next command every 0.5 seconds
        if time.time() - last_cmd_time > 0.5 and cmd_idx < len(commands):
            cmd, mag = commands[cmd_idx]
            sim.process_command(cmd, mag)
            print(f"[SIM] {cmd} ({mag:.1f})")
            cmd_idx += 1
            last_cmd_time = time.time()
        
        time.sleep(0.03)  # ~30 fps
    
    cv2.destroyAllWindows()