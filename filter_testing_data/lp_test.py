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
                positions.append([x, y, z])
    
    positions = np.array(positions)
    
    # Generate timestamps (assuming 30fps)
    timestamps = np.arange(len(positions)) / 30.0
    
    return positions, timestamps


def calculate_statistics(positions, label="Data"):
    """Calculate comprehensive statistics for position data"""
    
    # Basic statistics
    mean = np.mean(positions, axis=0)
    std = np.std(positions, axis=0)
    min_val = np.min(positions, axis=0)
    max_val = np.max(positions, axis=0)
    range_val = max_val - min_val
    
    # Jitter (frame-to-frame variation)
    if len(positions) > 1:
        frame_diff = np.diff(positions, axis=0)
        jitter = np.std(frame_diff, axis=0)
        max_jitter = np.max(np.abs(frame_diff), axis=0)
        mean_jitter = np.mean(np.abs(frame_diff), axis=0)
    else:
        jitter = np.zeros(3)
        max_jitter = np.zeros(3)
        mean_jitter = np.zeros(3)
    
    # Overall metrics
    total_std = np.linalg.norm(std)
    total_jitter = np.linalg.norm(jitter)
    
    return {
        'label': label,
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'jitter': jitter,
        'max_jitter': max_jitter,
        'mean_jitter': mean_jitter,
        'total_std': total_std,
        'total_jitter': total_jitter
    }


def print_statistics(stats):
    """Pretty print statistics"""
    print(f"\n{'='*70}")
    print(f"  {stats['label'].upper()}")
    print(f"{'='*70}")
    
    print(f"\n📊 Position Statistics (cm):")
    print(f"  Mean:   X={stats['mean'][0]:+7.3f}  Y={stats['mean'][1]:+7.3f}  Z={stats['mean'][2]:+7.3f}")
    print(f"  Std:    X={stats['std'][0]:7.4f}  Y={stats['std'][1]:7.4f}  Z={stats['std'][2]:7.4f}")
    print(f"  Range:  X={stats['range'][0]:7.4f}  Y={stats['range'][1]:7.4f}  Z={stats['range'][2]:7.4f}")
    print(f"  Min:    X={stats['min'][0]:+7.3f}  Y={stats['min'][1]:+7.3f}  Z={stats['min'][2]:+7.3f}")
    print(f"  Max:    X={stats['max'][0]:+7.3f}  Y={stats['max'][1]:+7.3f}  Z={stats['max'][2]:+7.3f}")
    
    print(f"\n📈 Jitter Analysis (frame-to-frame variation, cm):")
    print(f"  Std:    X={stats['jitter'][0]:7.4f}  Y={stats['jitter'][1]:7.4f}  Z={stats['jitter'][2]:7.4f}")
    print(f"  Mean:   X={stats['mean_jitter'][0]:7.4f}  Y={stats['mean_jitter'][1]:7.4f}  Z={stats['mean_jitter'][2]:7.4f}")
    print(f"  Max:    X={stats['max_jitter'][0]:7.4f}  Y={stats['max_jitter'][1]:7.4f}  Z={stats['max_jitter'][2]:7.4f}")
    
    print(f"\n🎯 Overall Metrics:")
    print(f"  Total Std Dev:    {stats['total_std']:.4f} cm")
    print(f"  Total Jitter:     {stats['total_jitter']:.4f} cm")


def compare_filters(raw_stats, filtered_stats_list):
    """Compare raw vs filtered performance"""
    
    print(f"\n{'='*70}")
    print(f"  FILTER COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Filter':<20} {'Std X':<10} {'Std Y':<10} {'Std Z':<10} {'Total':<10} {'Reduction'}")
    print(f"{'-'*70}")
    
    # Raw baseline
    print(f"{'Raw (baseline)':<20} {raw_stats['std'][0]:<10.4f} {raw_stats['std'][1]:<10.4f} "
          f"{raw_stats['std'][2]:<10.4f} {raw_stats['total_std']:<10.4f} {'—'}")
    
    # Filtered results
    for fstats in filtered_stats_list:
        reduction = (1 - fstats['total_std'] / raw_stats['total_std']) * 100
        print(f"{fstats['label']:<20} {fstats['std'][0]:<10.4f} {fstats['std'][1]:<10.4f} "
              f"{fstats['std'][2]:<10.4f} {fstats['total_std']:<10.4f} {reduction:>6.1f}%")
    
    print(f"\n{'Filter':<20} {'Jitter X':<10} {'Jitter Y':<10} {'Jitter Z':<10} {'Total':<10} {'Reduction'}")
    print(f"{'-'*70}")
    
    # Raw baseline
    print(f"{'Raw (baseline)':<20} {raw_stats['jitter'][0]:<10.4f} {raw_stats['jitter'][1]:<10.4f} "
          f"{raw_stats['jitter'][2]:<10.4f} {raw_stats['total_jitter']:<10.4f} {'—'}")
    
    # Filtered results
    for fstats in filtered_stats_list:
        reduction = (1 - fstats['total_jitter'] / raw_stats['total_jitter']) * 100
        print(f"{fstats['label']:<20} {fstats['jitter'][0]:<10.4f} {fstats['jitter'][1]:<10.4f} "
              f"{fstats['jitter'][2]:<10.4f} {fstats['total_jitter']:<10.4f} {reduction:>6.1f}%")


def test_filters(filename):
    """Test all three filters on the data"""
    
    positions, timestamps = load_thumb_data(filename)
    
    print(f"\n{'='*70}")
    print(f"  DATA LOADING")
    print(f"{'='*70}")
    print(f"  File:      {filename}")
    print(f"  Samples:   {len(positions)}")
    print(f"  Duration:  {timestamps[-1]:.2f} seconds")
    print(f"  FPS:       ~{len(positions)/timestamps[-1]:.1f}")
    
    # Calculate raw statistics
    raw_stats = calculate_statistics(positions, "Raw Data")
    print_statistics(raw_stats)
    
    # Initialize filters
    velocity_filter = VelocityBasedLowPassFilter(
        base_alpha=0.3,
        velocity_threshold=2.0,
        max_velocity=30.0
    )
    
    one_euro_filter = OneEuroFilter(
        min_cutoff=1.0,
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
    
    # Calculate filtered statistics
    vel_stats = calculate_statistics(velocity_filtered, "Velocity Filter")
    euro_stats = calculate_statistics(euro_filtered, "One Euro Filter")
    ema_stats = calculate_statistics(ema_filtered, "Simple EMA")
    
    print_statistics(vel_stats)
    print_statistics(euro_stats)
    print_statistics(ema_stats)
    
    # Compare all filters
    compare_filters(raw_stats, [vel_stats, euro_stats, ema_stats])
    
    # Benchmark verdict
    print(f"\n{'='*70}")
    print(f"  BENCHMARK VERDICT (Stationary Hand Test)")
    print(f"{'='*70}")
    
    acceptable_std = 0.5  # cm
    acceptable_jitter = 0.3  # cm
    
    print(f"\n✅ PASS Criteria:")
    print(f"  - Total Std Dev < {acceptable_std} cm")
    print(f"  - Total Jitter < {acceptable_jitter} cm")
    
    print(f"\n📋 Results:")
    for fstats in [vel_stats, euro_stats, ema_stats]:
        std_pass = "✅ PASS" if fstats['total_std'] < acceptable_std else "❌ FAIL"
        jitter_pass = "✅ PASS" if fstats['total_jitter'] < acceptable_jitter else "❌ FAIL"
        
        print(f"\n  {fstats['label']}:")
        print(f"    Std Dev: {fstats['total_std']:.4f} cm  {std_pass}")
        print(f"    Jitter:  {fstats['total_jitter']:.4f} cm  {jitter_pass}")
    
    print(f"\n{'='*70}\n")
    
    return {
        'timestamps': timestamps,
        'raw': positions,
        'velocity': velocity_filtered,
        'euro': euro_filtered,
        'ema': ema_filtered,
        'stats': {
            'raw': raw_stats,
            'velocity': vel_stats,
            'euro': euro_stats,
            'ema': ema_stats
        }
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
    # results = test_filters('filter_testing_data/thumb_tracking.txt')
    results = test_filters('filter_testing_data/monocular_tracking_data.txt')
    # results = test_filters('filter_testing_data/hand_keypoints_pinky.txt')

    plot_results(results)