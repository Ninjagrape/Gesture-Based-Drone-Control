# Gesture-Based Drone Control

A real-time computer vision system for controlling drones using hand gestures, combining monocular depth estimation, hand pose tracking, and adaptive filtering for intuitive 3D navigation.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-latest-orange.svg)

## Overview

This project implements a contactless drone control interface using computer vision and machine learning. By tracking hand position and orientation in 3D space, users can control drone movement through natural gestures without physical controllers.

### Key Features

- **Custom two stage landmarker model**: A custom first stage ROI detector using traditional segmentation models. Lighterweight.
- **3D Hand Tracking**: Monocular depth estimation from single camera using hand geometry
- **Gesture Recognition**: KNN-based classifier for static and dynamic gesture recognition
- **Orientation Tracking**: Full 6-DOF hand orientation (roll, pitch, yaw) for rotation control
- **Adaptive Filtering**: One Euro Filter and velocity-based low-pass filters for smooth, jitter-free tracking
- **Real-time Visualization**: 3D drone simulator with live position and orientation feedback
- **Multi-modal Control**: Separate modes for translation, rotation, and speed control

## System Architecture

```
┌─────────────────────┐
│   Camera Input      │
│   (640×480, 30fps)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│     MediaPipe Hand Landmarker       │
|        or custom two stage          |
│   (21 keypoints per hand, 2 hands)  │
└──────────┬──────────────────────────┘
           │
           ├──────────────────────────┐
           ▼                          ▼
┌──────────────────────┐   ┌──────────────────────┐
│  Gesture Recognition │   │  3D Position Tracker │
│  (KNN Classifier)    ├──►│  (Depth Estimation)  │
│  • Static gestures   │   │  • Multi-bone method │
│  • Dynamic gestures  │   │  • One Euro Filter   │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           ├──────────────────────────└─────────────┐
           ▼                                        │
┌──────────────────────────────────────────────┐    │
│       Hand Orientation Tracker               │    │
│  (Roll, Pitch, Yaw from Rotation Matrix)     │    │
└──────────┬───────────────────────────────────┘    │
           │                          ┌─────────────┘
           ▼                          ▼
┌──────────────────────────────────────────────┐
│         Control State Machine                │
│  • Idle → Translating → Idle                 │
│  • Idle → Rotating → Idle                    │
│  • Idle → Speed Control → Idle               │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│           3D Drone Simulator                 │
│  • 6-DOF motion simulation                   │
│  • Real-time visualization                   │
│  • Trail rendering                           │
└──────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or external camera
- Operating System: Linux, macOS, or Windows

### Dependencies

Install required packages:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn
```

### MediaPipe Model

Download the MediaPipe hand landmark model:

```bash
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Place `hand_landmarker.task` in the project root directory.

## Usage

### Quick Start

Run the main control system:

```bash
python knn_hand_tracking.py
```

### Gesture Commands

The system operates in different control modes:

#### Entry Gestures (Static)
- **ACCELERATE** (Closed fist ✊): Enter acceleration mode
- **ROTATE** (devil horns 🤙): Enter rotation mode
- **TRANSLATION** (Two fingers pointed in any comfortable direction👆): Enter motion mode
- **GO** (pinch 🤏): Release drone from safemode

#### Exit Gestures (Static)
- **STOP** (open palm, all 5 fingers extended ✋ ): Exit current mode and return to idle
- **HALT** (Fingers curved into a C): Exit current mode and return to idle

#### Translation Commands (3D Position)
- **Move hand left/right**: Lateral translation (>3cm threshold)
- **Move hand up/down**: Vertical translation (>3cm threshold)
- **Move hand forward/back**: Depth translation (>4cm threshold)

#### Rotation Commands (Hand Orientation)
- **Roll wrist left/right**: Roll control (>10° threshold)
- **Rotate arm**: Yaw control (>15° threshold)
- **Flop wrist up/down**: Pitch control (>10° threshold)

#### Speed Control Commands
- **Move hand forward**: Accelerate
- **Move hand backward**: Decelerate

### Keyboard Controls

- **Q**: Quit application
- **R**: Record gesture samples (for training)
- **F**: Reset position filter
- **S**: Set reference position (for debugging)
- **T**: Toggle between Mediapipe integrated two stage and custom two stage model
- **1-5**: Select gesture label for recording

### Drone Simulator Controls

- **Arrow Keys**: Rotate view
- **+/-**: Zoom in/out
- **R**: Reset drone position
- **Q**: Close simulator

## Technical Details

### Two-Stage Segmentation Model

The system uses a hybrid detection approach combining classical computer vision with MediaPipe:

**Stage 1 - Palm Region Detection**:
- **Skin Segmentation**: Dual HSV + YCrCb color space filtering with adaptive thresholding
- **Face Exclusion**: Haar Cascade classifier removes false positives from facial regions
- **Contour Analysis**: Detects palm via vertical projection drop-off at wrist boundary
- **Rotated Bounding Box**: Fits minimal area rectangle to palm contour, normalized to upright orientation

**Stage 2 - MediaPipe Landmark Refinement**:
- Extracts ROI from Stage 1 (256×256px upright crop)
- Runs MediaPipe Hands on isolated palm region
- Maps landmarks back to original frame via inverse affine transformation
- Fallback to full-frame detection if palm segmentation fails

**Tracking**: IoU-based multi-hand tracker with One Euro filtering on bounding boxes (position: β=0.015, size: β=0.001)

### Monocular Depth Estimation

The system estimates 3D hand position from a single camera using geometric constraints:

**Method**: Multi-bone depth estimation
- Uses 7 hand bone segments (wrist-to-MCP, proximal phalanges)
- Computes depth via similar triangles: `Z = (f × L_real) / L_pixel`
- Weighted averaging with outlier removal (MAD-based)
- Focal length calibration: `f ≈ frame_width` pixels

**Accuracy**: ±2-3cm at 30-100cm distance

### Gesture Recognition

**Algorithm**: K-Nearest Neighbors (KNN) with feature engineering

**Features** 
- 74 + 3 (velocity) dimensions
- Normalized 21-landmark positions (63D)
- Pairwise distances between landmarks (21×20/2 = 210 pairs → reduced)
- Finger extension state (5D)
- Palm orientation normal vector (3D)

***Static Dataset***
- samples=19259 
- features_dims=74 
- classes=['CHANGE SPEED', 'GO', 'HOLD', 'ROTATE', 'STOP', 'TRANSLATE']
  
***Dynamic Dataset***
- samples=293 
- feature_dims=77 
- classes=['CHANGE SPEED', 'GO', 'HOLD', 'ROTATE', 'STOP', 'TRANSLATE']

**Classification**:
- Static gestures: Spatial features only
- Dynamic gestures: Spatial + velocity features (3D motion vector)

**Training**: Interactive recording with real-time feedback

### Low-Pass Filtering

Two complementary filtering approaches:

#### One Euro Filter
- Adaptive cutoff frequency based on signal velocity
- Parameters: `min_cutoff=1.0Hz, beta=0.007, d_cutoff=1.0Hz`
- Used for: Position tracking, orientation smoothing

#### Velocity-Based Low-Pass Filter
- Smoothing adapts to hand speed
- Fast motion → more responsive (α≈1.0)
- Slow motion → more stable (α≈0.05)
- Used for: High-frequency jitter reduction

### Orientation Tracking

**Method**: Rotation matrix decomposition from hand geometry

1. Construct hand coordinate frame:
   - X-axis: Thumb → Pinky
   - Z-axis: Palm normal (cross product)
   - Y-axis: Wrist → Fingers (orthogonalized)

2. Extract Euler angles via XYZ decomposition:
   - Roll: Wrist rotation around finger axis
   - Pitch: Wrist flexion/extension
   - Yaw: Arm rotation in horizontal plane

3. Single-axis control:
   - Detects dominant axis (largest angular change)
   - Prevents simultaneous multi-axis commands
   - Deadzone: 10° (roll/pitch), 15° (yaw)


## Project Structure

```
gesture-drone-control/
├── knn_hand_tracking.py           # Main control system
├── monocular_depth_tracking.py    # 3D position estimation
├── orientation_tracking.py        # Hand orientation tracking
├── two_stage_detection.py         # Custom two stage model
├── lp_filt.py                     # Low-pass filter implementations
├── drone_simulator.py             # 3D visualization
├── hand_landmarker.task           # MediaPipe model (download separately)
├── gesture_static.pkl             # Trained KNN model (generated)
├── gesture_dynamic.pkl            # Trained KNN model (generated)
└── README.md                      # This file
```

## Configuration

### Depth Estimation Parameters

```python
# In monocular_depth_tracking.py
REAL_HAND_LENGTH = 19.0  # cm (adult average)
TRANSLATE_THRESHOLD = 3.0  # cm
YAW_THRESHOLD = 5.0  # cm
ACCEL_THRESHOLD = 4.0  # cm
```

### Orientation Thresholds

```python
# In orientation_tracking.py
ROLL_THRESHOLDS = {'small': 10°, 'medium': 20°, 'large': 35°}
YAW_THRESHOLDS = {'small': 15°, 'medium': 30°, 'large': 50°}
PITCH_THRESHOLDS = {'small': 10°, 'medium': 20°, 'large': 35°}
DEADZONE = 10.0  # degrees
```

### Filter Parameters

```python
# One Euro Filter
min_cutoff = 1.0   # Hz
beta = 0.007       # Sensitivity to velocity
d_cutoff = 1.0     # Hz (derivative smoothing)

# Velocity-Based Filter
base_alpha = 0.3
velocity_threshold = 5.0   # cm/s
max_velocity = 50.0        # cm/s
```

## Troubleshooting

### Hand Not Detected
- Ensure good lighting conditions
- Keep hand 30-100cm from camera
- Face palm toward camera for better detection
- Avoid cluttered backgrounds

### Jittery Tracking
- Adjust filter parameters (decrease `min_cutoff` for more smoothing)
- Ensure stable camera mounting
- Check for motion blur (reduce exposure time)

### Incorrect Gestures
- Retrain KNN model with more samples (press 'R' to record)
- Increase confidence threshold (adjust `MIN_STATIC_CONF`)
- Ensure clear finger extension/flexion

### Depth Estimation Issues
- Calibrate focal length for your camera
- Check lighting (depth estimation requires visible hand features)
- Adjust `REAL_HAND_LENGTH` for hand size

## Future Improvements

### Planned Features
- [ ] Deep learning-based gesture recognition
- [ ] Multi-hand coordination gestures
- [ ] Real drone hardware integration

### Research Directions
- [ ] Hand-object interaction detection
- [ ] Adaptive thresholding based on user behaviour
- [ ] Temporal gesture sequences (gesture phrases)
- [ ] Transfer learning from pre-trained models
- [ ] Real-time performance optimisation (GPU acceleration)

## References

### Academic Papers
1. Casiez, G., Roussel, N. and Vogel, D. (2012). "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"


### Libraries and Tools
- [MediaPipe](https://mediapipe.dev/) - Google's ML framework for hand tracking
- [OpenCV](https://opencv.org/) - Computer vision library
- [scikit-learn](https://scikit-learn.org/) - Machine learning toolkit

## Authors

Morris Lee
The University of Sydney

Marinelle Juan
The University of Sydney

Jarvis Du
The University of Sydney

## License

GNU GENERAL PUBLIC LICENSE v3

## Acknowledgments

- MediaPipe team for hand landmark detection
- Dr. Mitch Bryson for invaluable tutelege and advice
- Nikolai Goncherov for invaluable advice


---

**Note**: This is a research/educational project. For production drone control systems, additional safety features, redundancy, and regulatory compliance would be required.
