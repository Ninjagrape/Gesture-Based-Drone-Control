"""
lp_filt.py - Low-Pass Filters for 3D Hand Tracking

Usage:
    from lp_filt import VelocityBasedLowPassFilter, OneEuroFilter
    
    filter = VelocityBasedLowPassFilter()
    filtered_pos = filter.update(raw_position)
"""

import numpy as np
import time
from collections import deque


class VelocityBasedLowPassFilter:
    """
    Low-pass filter that adapts smoothing based on hand velocity
    
    Key idea: 
    - Fast movements → Less smoothing (more responsive)
    - Slow movements → More smoothing (reduces jitter)
    - Stationary hand → Maximum smoothing (very stable)
    """
    
    def __init__(self, 
                 base_alpha=0.3,           # Base smoothing factor (0-1)
                 velocity_threshold=5.0,    # Velocity in cm/s for full smoothing
                 max_velocity=50.0,         # Velocity for minimum smoothing
                 history_size=5):           # Frames to calculate velocity
        """
        Initialize velocity-based low-pass filter
        
        Args:
            base_alpha: Base smoothing (0=no smoothing, 1=all measurement)
            velocity_threshold: Below this velocity, use maximum smoothing
            max_velocity: Above this velocity, use minimum smoothing
            history_size: Number of frames for velocity calculation
        """
        self.base_alpha = base_alpha
        self.velocity_threshold = velocity_threshold
        self.max_velocity = max_velocity
        
        # State
        self.filtered_position = None
        self.position_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=10)
        
        self.current_velocity = 0.0
        self.last_update_time = None
        
    def calculate_velocity(self):
        """Calculate current velocity from position history"""
        if len(self.position_history) < 2:
            return 0.0
        
        positions = np.array(list(self.position_history))
        times = np.array(list(self.time_history))
        
        dt = times[-1] - times[0]
        
        if dt < 0.001:  # Avoid division by zero
            return self.current_velocity
        
        dp = positions[-1] - positions[0]
        velocity_vector = dp / dt
        velocity_magnitude = np.linalg.norm(velocity_vector)
        
        return velocity_magnitude
    
    def adaptive_alpha(self, velocity):
        """
        Calculate adaptive smoothing factor based on velocity
        
        Returns alpha in range [0.05, 1.0]:
        - Low velocity → alpha ≈ 0.05 (heavy smoothing)
        - High velocity → alpha ≈ 1.0 (minimal smoothing, responsive)
        """
        if velocity < self.velocity_threshold:
            alpha = 0.05
        elif velocity > self.max_velocity:
            alpha = 1.0
        else:
            ratio = (velocity - self.velocity_threshold) / (self.max_velocity - self.velocity_threshold)
            alpha = 0.05 + ratio * 0.95
        
        return alpha
    
    def update(self, measurement, timestamp=None):
        """
        Update filter with new measurement
        
        Args:
            measurement: [x, y, z] position in cm (can be numpy array or list)
            timestamp: optional timestamp (uses current time if None)
            
        Returns:
            filtered position as numpy array [x, y, z]
        """
        measurement = np.array(measurement, dtype=float)
        
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize on first measurement
        if self.filtered_position is None:
            self.filtered_position = measurement.copy()
            self.position_history.append(measurement)
            self.time_history.append(timestamp)
            self.last_update_time = timestamp
            return self.filtered_position.copy()
        
        # Add to history
        self.position_history.append(measurement)
        self.time_history.append(timestamp)
        
        # Calculate velocity
        velocity = self.calculate_velocity()
        self.velocity_history.append(velocity)
        self.current_velocity = velocity
        
        # Get adaptive alpha based on velocity
        alpha = self.adaptive_alpha(velocity)
        
        # Apply exponential moving average
        self.filtered_position = alpha * measurement + (1 - alpha) * self.filtered_position
        
        self.last_update_time = timestamp
        
        return self.filtered_position.copy()
    
    def get_velocity(self):
        """Get current velocity magnitude in cm/s"""
        return self.current_velocity
    
    def get_velocity_vector(self):
        """Get current velocity as vector [vx, vy, vz]"""
        if len(self.position_history) < 2:
            return np.array([0.0, 0.0, 0.0])
        
        positions = np.array(list(self.position_history))
        times = np.array(list(self.time_history))
        
        dt = times[-1] - times[0]
        if dt < 0.001:
            return np.array([0.0, 0.0, 0.0])
        
        dp = positions[-1] - positions[0]
        return dp / dt
    
    def get_smoothed_velocity(self):
        """Get smoothed velocity (average over recent history)"""
        if len(self.velocity_history) == 0:
            return 0.0
        return np.mean(list(self.velocity_history))
    
    def reset(self):
        """Reset filter state"""
        self.filtered_position = None
        self.position_history.clear()
        self.time_history.clear()
        self.velocity_history.clear()
        self.current_velocity = 0.0


class OneEuroFilter:
    """
    One Euro Filter - sophisticated low-pass filter used in real-time tracking
    
    Automatically adapts cutoff frequency based on velocity
    Used in VR controllers, motion capture, and hand tracking
    
    Reference: Casiez, G., Roussel, N. and Vogel, D. (2012). 
    "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
    """
    
    def __init__(self, 
                 min_cutoff=1.0,      # Minimum cutoff frequency (Hz)
                 beta=0.007,          # Cutoff slope (sensitivity to velocity)
                 d_cutoff=1.0):       # Cutoff for velocity derivative
        """
        Initialize One Euro Filter
        
        Args:
            min_cutoff: Minimum cutoff frequency (lower = more smoothing)
            beta: How much velocity affects smoothing (higher = more responsive)
            d_cutoff: Derivative cutoff (smoothing for velocity calculation)
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
        
    def smoothing_factor(self, t_e, cutoff):
        """Calculate smoothing factor from cutoff frequency"""
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)
    
    def exponential_smoothing(self, alpha, x, x_prev):
        """Apply exponential smoothing"""
        return alpha * x + (1 - alpha) * x_prev
    
    def update(self, x, timestamp=None):
        """
        Update filter with new measurement
        
        Args:
            x: measurement (can be scalar, list, or numpy array)
            timestamp: time in seconds
            
        Returns:
            filtered value as numpy array
        """
        x = np.array(x, dtype=float)
        
        if timestamp is None:
            timestamp = time.time()
        
        # Initialize
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = timestamp
            return x.copy()
        
        # Calculate time difference
        t_e = timestamp - self.t_prev
        
        if t_e < 0.0001:  # Avoid division by zero
            return self.x_prev.copy()
        
        # Calculate derivative (velocity)
        dx = (x - self.x_prev) / t_e
        
        # Smooth derivative
        alpha_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx_hat = self.exponential_smoothing(alpha_d, dx, self.dx_prev)
        
        # Adaptive cutoff frequency based on velocity
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        
        # Smooth signal
        alpha = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(alpha, x, self.x_prev)
        
        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = timestamp
        
        return x_hat.copy()
    
    def get_velocity(self):
        """Get current velocity estimate"""
        if self.dx_prev is None:
            return 0.0
        return np.linalg.norm(self.dx_prev)
    
    def reset(self):
        """Reset filter"""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class SimpleEMA:
    """Simple exponential moving average (for comparison/baseline)"""
    
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha: Smoothing factor (0-1)
                   0 = maximum smoothing, 1 = no smoothing
        """
        self.alpha = alpha
        self.value = None
        
    def update(self, measurement):
        """Update with new measurement"""
        measurement = np.array(measurement, dtype=float)
        if self.value is None:
            self.value = measurement
        else:
            self.value = self.alpha * measurement + (1 - self.alpha) * self.value
        return self.value.copy()
    
    def reset(self):
        """Reset filter"""
        self.value = None