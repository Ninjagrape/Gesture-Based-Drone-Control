"""
lp_test.py - Test low-pass filters on stationary hand data

Tests different filters on real MediaPipe hand tracking data
and visualizes the smoothing effect.
"""

import numpy as np
import matplotlib.pyplot as plt
import re

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lp_filt import VelocityBasedLowPassFilter, OneEuroFilter, SimpleEMA

def test_filters_rot(filename, keypoint='Wrist'):
    # Regex to extract "[x, y, z]" after the given keypoint label
    pattern = re.compile(rf'{keypoint}:\s*\[([^\]]+)\]')

    data = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            match = pattern.search(line)
            if not match:
                continue  # skip if that keypoint wasn't detected in this frame

            coords = [float(x.strip()) for x in match.group(1).split(',')]
            data.append(coords)

    data = np.array(data)
    print(f"Loaded {len(data)} frames of '{keypoint}' keypoint from {filename}")
    return data


def load_thumb_data(filename):
    """Load thumb tracking data from CSV file"""
    positions = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse CSV values
            values = line.split(',')
            if len(values) == 3:
                x, y, z = [float(v.strip()) for v in values]
                # Convert to cm (assuming data is in meters)
                positions.append([x * 100, y * 100, z * 100])
    
    positions = np.array(positions)
    
    # Generate timestamps (assuming 30fps)
    timestamps = np.arange(len(positions)) / 30.0
    
    return positions, timestamps


def test_filters(filename):
    """Test all three filters on the data"""
    
    positions, timestamps = load_thumb_data(filename)
    
    print(f"Loaded {len(positions)} thumb positions")
    print(f"Duration: {timestamps[-1]:.2f} seconds")
    
    # Calculate raw jitter statistics
    raw_std = np.std(positions, axis=0)
    print(f"\nRaw data std dev: X={raw_std[0]:.4f}, Y={raw_std[1]:.4f}, Z={raw_std[2]:.4f} cm")
    
    # Initialize filters
    velocity_filter = VelocityBasedLowPassFilter(
        base_alpha=0.3,
        velocity_threshold=2.0,
        max_velocity=30.0
    )
    
    one_euro_filter = OneEuroFilter(
        min_cutoff=1.0,
        # beta=0.007,
        beta=0.05,
        d_cutoff=1.0
    )
    
    simple_ema = SimpleEMA(alpha=0.3)
    
    # Apply filters
    velocity_filtered = []
    euro_filtered = []
    ema_filtered = []
    
    for i, (pos, t) in enumerate(zip(positions, timestamps)):
        velocity_filtered.append(velocity_filter.update(pos, t))
        euro_filtered.append(one_euro_filter.update(pos, t))
        ema_filtered.append(simple_ema.update(pos))
    
    velocity_filtered = np.array(velocity_filtered)
    euro_filtered = np.array(euro_filtered)
    ema_filtered = np.array(ema_filtered)
    
    # Calculate filtered jitter statistics
    vel_std = np.std(velocity_filtered, axis=0)
    euro_std = np.std(euro_filtered, axis=0)
    ema_std = np.std(ema_filtered, axis=0)
    
    print(f"\nVelocity Filter std dev: X={vel_std[0]:.4f}, Y={vel_std[1]:.4f}, Z={vel_std[2]:.4f} cm")
    print(f"One Euro Filter std dev: X={euro_std[0]:.4f}, Y={euro_std[1]:.4f}, Z={euro_std[2]:.4f} cm")
    print(f"Simple EMA std dev: X={ema_std[0]:.4f}, Y={ema_std[1]:.4f}, Z={ema_std[2]:.4f} cm")
    
    print(f"\nJitter reduction:")
    print(f"Velocity Filter: {(1 - np.mean(vel_std) / np.mean(raw_std)) * 100:.1f}%")
    print(f"One Euro Filter: {(1 - np.mean(euro_std) / np.mean(raw_std)) * 100:.1f}%")
    print(f"Simple EMA: {(1 - np.mean(ema_std) / np.mean(raw_std)) * 100:.1f}%")
    
    return {
        'timestamps': timestamps,
        'raw': positions,
        'velocity': velocity_filtered,
        'euro': euro_filtered,
        'ema': ema_filtered
    }


def plot_results(results):
    """Create visualization of filter performance"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Low-Pass Filter Comparison - Thumb Tracking', fontsize=16, fontweight='bold')
    
    timestamps = results['timestamps']
    colors = {
        'raw': '#FF6B6B',
        'velocity': '#4ECDC4',
        'euro': '#95E1D3',
        'ema': '#FFA07A'
    }
    
    axis_names = ['X', 'Y', 'Z']
    
    for i, (ax, axis_name) in enumerate(zip(axes, axis_names)):
        # Plot raw data
        ax.plot(timestamps, results['raw'][:, i], 
                color=colors['raw'], alpha=0.4, linewidth=1, 
                label='Raw (Noisy)', marker='o', markersize=3, markevery=3)
        
        # Plot filtered data
        ax.plot(timestamps, results['velocity'][:, i], 
                color=colors['velocity'], linewidth=2, 
                label='Velocity Filter')
        
        ax.plot(timestamps, results['euro'][:, i], 
                color=colors['euro'], linewidth=2, 
                label='One Euro Filter', linestyle='--')
        
        ax.plot(timestamps, results['ema'][:, i], 
                color=colors['ema'], linewidth=2, 
                label='Simple EMA', linestyle=':')
        
        # Formatting
        ax.set_ylabel(f'{axis_name} Position (cm)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        
        # Add std dev text
        raw_std = np.std(results['raw'][:, i])
        vel_std = np.std(results['velocity'][:, i])
        
        ax.text(0.02, 0.98, f'Raw σ: {raw_std:.4f} cm\nFiltered σ: {vel_std:.4f} cm', 
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Second figure: 3D trajectory
    fig2 = plt.figure(figsize=(12, 8))
    ax3d = fig2.add_subplot(111, projection='3d')
    
    ax3d.plot(results['raw'][:, 0], results['raw'][:, 1], results['raw'][:, 2],
              color=colors['raw'], alpha=0.4, linewidth=1, label='Raw')
    
    ax3d.plot(results['velocity'][:, 0], results['velocity'][:, 1], results['velocity'][:, 2],
              color=colors['velocity'], linewidth=2, label='Velocity Filter')
    
    ax3d.plot(results['euro'][:, 0], results['euro'][:, 1], results['euro'][:, 2],
              color=colors['euro'], linewidth=2, label='One Euro Filter', linestyle='--')
    
    ax3d.set_xlabel('X (cm)', fontweight='bold')
    ax3d.set_ylabel('Y (cm)', fontweight='bold')
    ax3d.set_zlabel('Z (cm)', fontweight='bold')
    ax3d.set_title('3D Thumb Position Tracking', fontsize=14, fontweight='bold')
    ax3d.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load data from file

    results = test_filters('filter_testing_data/thumb_tracking.txt')
    # results = test_filters('filter_testing_data/hand_keypoints_pinky.txt')

    plot_results(results)