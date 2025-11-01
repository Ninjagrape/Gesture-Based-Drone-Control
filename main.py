##this is the messiest absolute AI slop of an integration I gave Claude to just see if things would work. It does but 
##for your own sanity do not look down this is absolute utter garbage

#LIBRARIES----------------------------------------------------------------------------------------------------------------------------------------------
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pandas as pd
from collections import deque
import pickle 
import os
import math

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Import 3D tracking filter
from lp_filt import OneEuroFilter

#DRONE DICT----------------------------------------------------------------------------

'''drone gesture dictionary
STOP

GO

HOLD

CHANGE SPEED
    ACCELERATE
    DECLERATE 

TRANSLATE
    MOVE UP
    MOVE DOWN
    MOVE LEFT
    MOVE RIGHT

ROTATE 
    ROLL
    PITCH
    YAW (ROTATE)
'''

#LANDMARK GLOBALS--------------------------------------------------------------------------------------
LANDMARKS = {
    0:  "WRIST",
    1:  "THUMB_CMC",
    2:  "THUMB_MCP",
    3:  "THUMB_IP",
    4:  "THUMB_TIP",
    5:  "INDEX_MCP",
    6:  "INDEX_PIP",
    7:  "INDEX_DIP",
    8:  "INDEX_TIP",
    9:  "MIDDLE_MCP",
    10: "MIDDLE_PIP",
    11: "MIDDLE_DIP",
    12: "MIDDLE_TIP",
    13: "RING_MCP",
    14: "RING_PIP",
    15: "RING_DIP",
    16: "RING_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}

WRIST = 0
THUMB_FINGER  = [1, 2, 3, 4]
INDEX_FINGER  = [5, 6, 7, 8]
MIDDLE_FINGER = [9, 10, 11, 12]
RING_FINGER   = [13, 14, 15, 16]
PINKY_FINGER  = [17, 18, 19, 20]

FINGERTIPS = [4, 8, 12, 16, 20]
MCP        = [2, 5, 9, 13, 17]  # base knuckles
IP         = [3]
CMC        = [1]
PIP        = [6, 10, 14, 18]  # middle joints

# named indices
THUMB, INDEX, MIDDLE, RING, PINKY = 0, 1, 2, 3, 4


#3D TRACKING INTEGRATION----------------------------------------------------------------
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
                return np.average(depth_array, weights=weight_array)
        
        return 100.0  # Default depth
    
    def get_wrist_position_3d(self, hand_landmarks, frame_shape):
        """Get filtered 3D wrist position"""
        h, w = frame_shape[:2]
        wrist = hand_landmarks[WRIST]
        
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
        print(f"[3D] Reference set: X={position_3d[0]:.1f}, Y={position_3d[1]:.1f}, Z={position_3d[2]:.1f} cm")
    
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
            if dx > 0:
                commands.append(("MOVE RIGHT", abs(dx)))
            else:
                commands.append(("MOVE LEFT", abs(dx)))
        
        if abs(dy) > self.TRANSLATE_THRESHOLD:
            if dy > 0:
                commands.append(("MOVE DOWN", abs(dy)))
            else:
                commands.append(("MOVE UP", abs(dy)))
        
        # Yaw commands (X axis with higher threshold)
        if abs(dx) > self.YAW_THRESHOLD:
            if dx > 0:
                commands.append(("YAW RIGHT", abs(dx)))
            else:
                commands.append(("YAW LEFT", abs(dx)))
        
        # Acceleration commands (Z axis - depth)
        if abs(dz) > self.ACCEL_THRESHOLD:
            if dz < 0:  # Moving towards camera
                commands.append(("ACCELERATE", abs(dz)))
            else:  # Moving away from camera
                commands.append(("DECELERATE", abs(dz)))
        
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
            'distance': distance_3d,
            'current': current_position,
            'reference': self.reference_position
        }
    
    def reset_filter(self):
        """Reset the position filter"""
        self.position_filter.reset()


#CLASSES------------------------------------------------------------------------------------
class KNNGesture:
    def __init__(self, n_neighbors=5):
        self.scaler = StandardScaler()
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        self.X = []
        self.y = []
        self.is_trained = False

    def add_sample(self, feats, label):
        self.X.append(feats.astype(np.float32))
        self.y.append(label)

    def fit(self):
        if len(self.X) < 2:
            print("[WARN] not enough samples to train")
            return
        X = np.stack(self.X, axis=0)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, np.array(self.y))
        self.is_trained = True
        print(f"[INFO] trained knn on {len(self.y)} samples, classes={sorted(set(self.y))}")

    def predict(self, feats):
        if not self.is_trained:
            return None, 0.0
        fs = self.scaler.transform(feats.reshape(1, -1))
        pred = self.clf.predict(fs)[0]
        proba = float(self.clf.predict_proba(fs).max()) if hasattr(self.clf, "predict_proba") else 0.0
        return pred, proba
    
    def save(self, path="gesture_knn.pkl"):
        """save trained scaler and classifier"""
        if not self.is_trained:
            print("[WARN] model not trained. nothing saved")
            return
        payload = {
            "scaler": self.scaler,
            "clf": self.clf,
            "feature_dim": self.clf._fit_X.shape[1] if hasattr(self.clf, "_fit_X") else None,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] saved model → {os.path.abspath(path)}")

    @classmethod
    def load(cls, path="gesture_knn.pkl"):
        """load scaler and classifier if file exists"""
        if not os.path.isfile(path):
            print(f"[INFO] no saved model at {path}")
            return None
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj.scaler = payload["scaler"]
        obj.clf = payload["clf"]
        obj.is_trained = True
        print(f"[INFO] loaded model ← {os.path.abspath(path)}")
        return obj

class GestureHands:
    def __init__(self):
        # one history per hand index (0/1)
        self.landmark_history = {0: [], 1: []}

    def track_gesture_motion(self, hand_index, hand_landmarks):
        """track hand motion over time for dynamic gestures"""
        hist = self.landmark_history[hand_index]

        # store wrist/palm center (landmark 0)
        palm_center = hand_landmarks[0]
        hist.append([palm_center.x, palm_center.y, getattr(palm_center, "z", 0.0)])

        # keep only recent history (≈ 1s @30fps)
        if len(hist) > 10:
            hist.pop(0)

        # calculate mean velocity over the window
        if len(hist) > 1:
            velocities = np.diff(np.asarray(hist, dtype=np.float32), axis=0)
            return np.mean(velocities, axis=0)  # shape (3,)
        return None


#DRAWING FUNCTIONS---------------------------------------------------------------------------
def draw_landmarks_on_image(image, detection_result):
    """ draw hand landmarks and connections on an rgb image (np array). """
    annotated_image = np.copy(image)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),
                (0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20)
            ]
            for start_idx, end_idx in connections:
                s = hand_landmarks[start_idx]; e = hand_landmarks[end_idx]
                x1 = int(s.x * annotated_image.shape[1]); y1 = int(s.y * annotated_image.shape[0])
                x2 = int(e.x * annotated_image.shape[1]); y2 = int(e.y * annotated_image.shape[0])
                cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated_image


def draw_gesture_label(image, hand_or_box, gesture, *, pad=12):
    """ draw a box around the hand with the gesture text inside. """
    if hand_or_box is None:
        return image
    h, w = image.shape[:2]

    # compute bounding box from normalised landmarks
    xs = [int(lm.x * w) for lm in hand_or_box]
    ys = [int(lm.y * h) for lm in hand_or_box]
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad)

    # clamp and draw
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    text_scale = 0.6
    text_th = 1
    tx = int(x1)
    ty = int(y1 - 0.05 * y1)
    cv2.putText(image, str(gesture), (tx, ty), font, text_scale, (0, 0, 255), text_th, cv2.LINE_AA)
    return image


#LANDMARK FUNCTIONS-------------------------------------------------------------------------------------------
def palm_facing_camera(result):
    hand = result.handedness[0][0].category_name  # "Right" or "Left"
    lm = result.hand_landmarks[0]

    v1 = np.array([lm[MCP[INDEX]].x - lm[0].x, 
                   lm[MCP[INDEX]].y - lm[0].y,
                   getattr(lm[MCP[INDEX]], 'z', 0.0) - getattr(lm[0], 'z', 0.0)])
    v2 = np.array([lm[MCP[PINKY]].x - lm[0].x,
                   lm[MCP[PINKY]].y - lm[0].y,
                   getattr(lm[MCP[PINKY]], 'z', 0.0) - getattr(lm[0], 'z', 0.0)])
    nz_palm = np.cross(v1,v2)[2]
    
    if hand == "Right":
        return nz_palm <= 0
    else:
        return nz_palm >= 0
        
def extract_features_from_hand(hand_landmarks):
    """ build a compact, scale/translation-invariant feature vector. """
    pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in hand_landmarks], dtype=np.float32)

    wrist = pts[WRIST].copy()
    rel = pts - wrist

    base_span = np.linalg.norm(pts[5] - pts[17]) or 1e-6
    rel /= base_span

    def angle(a, b, c):
        v1, v2 = a - b, c - b
        n1 = np.linalg.norm(v1) or 1e-6
        n2 = np.linalg.norm(v2) or 1e-6
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return np.arccos(cosang)

    thumb_ang = angle(rel[1],  rel[2],  rel[3])
    idx_ang   = angle(rel[0],  rel[5],  rel[6])
    mid_ang   = angle(rel[0],  rel[9],  rel[10])
    ring_ang  = angle(rel[0],  rel[13], rel[14])
    pinky_ang = angle(rel[0],  rel[17], rel[18])
    angles = np.array([thumb_ang, idx_ang, mid_ang, ring_ang, pinky_ang], dtype=np.float32)

    feats = np.concatenate([rel.flatten(), angles], axis=0)
    return feats


#MAIN()----------------------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize 3D tracker
wrist_tracker = Wrist3DTracker(frame_width=w)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("annotated.mp4", fourcc, fps, (w, h))

# recording + controls
STATIC_GESTURE_KEYS = {
    ord('1'): "STOP",
    ord('2'): "HOLD",
    ord('3'): "CHANGE SPEED",
    ord('4'): "TRANSLATE",
    ord('5'): "ROTATE"
}

MIN_STATIC_CONF = 0.85

# static gestures that switch/clear modes
MODE_GESTURES = {
    "CHANGE SPEED": "changing_speed",
    "TRANSLATE":    "translating",
    "ROTATE":       "rotating",
}
EXIT_GESTURES = {"STOP", "HOLD"}

current_label = "STOP"
N_SAMPLES = 100
recording = False
samples_left = 0

frame_idx = 0

knn_static  = KNNGesture.load("gesture_static.pkl")  or KNNGesture(n_neighbors=7)
knn_dynamic = KNNGesture.load("gesture_dynamic.pkl") or KNNGesture(n_neighbors=7)

fps_hist = deque(maxlen=30)

temporal_multi = GestureHands()

stable_label = None
gesture_stage = "idle"

print("\n")
print("-------------------------------------------------------")
print("3D GESTURE CONTROL SYSTEM")
print("-------------------------------------------------------")
print("HOW TO RUN:")
print("Press KEY[1-5] to set label for training")
print("Press KEY['r'] to record samples")
print("Press KEY['q'] to quit and save trained model/s")
print("\n")
print("GESTURE MODES:")
print("  STOP/HOLD    - Exit current mode")
print("  CHANGE SPEED - 3D depth control (forward=accel, back=decel)")
print("  TRANSLATE    - 3D movement (up/down/left/right)")
print("  ROTATE       - 3D yaw control (left/right movement)")
print("-------------------------------------------------------")
print("\n")


#MAIN LOOP--------------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_idx += 1

    # bgr->rgb for mediapipe tasks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # detect hands
    detection_result = detector.detect(mp_image)

    # base overlay (skeleton)
    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

    if detection_result.hand_landmarks:
        hands = list(detection_result.hand_landmarks)
        hands.sort(key=lambda hl: hl[WRIST].x)

        for hi, hand in enumerate(hands):
            # Get 3D position
            pos_3d = wrist_tracker.get_wrist_position_3d(hand, (h, w))
            
            # temporal motion for classification
            v = temporal_multi.track_gesture_motion(hi, hand)
            speed = np.linalg.norm(v) if v is not None else 0.0
            gesture_type = "dynamic" if speed > 0.005 else "static"

            # Extract features
            feats = extract_features_from_hand(hand)

            # Recording mode
            if recording and samples_left > 0:
                motion_feats = v if v is not None else np.zeros(3)
                speed = np.linalg.norm(v) if v is not None else 0.0
                gesture_type = "dynamic" if speed > 0.02 else "static"

                if gesture_type == "static":
                    knn_static.add_sample(feats, current_label)
                else:
                    combined = np.concatenate([feats, motion_feats])
                    knn_dynamic.add_sample(combined, current_label)

                samples_left -= 1
                cv2.putText(annotated_frame, f"rec {current_label}: {N_SAMPLES - samples_left}/{N_SAMPLES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                if samples_left == 0:
                    recording = False
                    knn_static.fit()
                    knn_dynamic.fit()
            
            # Prediction
            pred, conf = ("none", 0.0)
            
            if gesture_type == "static" and knn_static.is_trained:
                pred, conf = knn_static.predict(feats)

                if pred in ("STOP", "HOLD"):
                    if not palm_facing_camera(detection_result):
                        pred = "none"
                        conf = 0.0

                if conf < MIN_STATIC_CONF:
                    pred = "none"

            elif gesture_type == "dynamic" and knn_dynamic.is_trained:
                motion_feats = v if v is not None else np.zeros(3)
                combined = np.concatenate([feats, motion_feats])
                pred, conf = knn_dynamic.predict(combined)

            # GESTURE STATE MACHINE
            if gesture_type == "static" and pred != "none" and conf >= MIN_STATIC_CONF:
                if pred in EXIT_GESTURES:
                    gesture_stage = "idle"
                    stable_label = pred
                    wrist_tracker.clear_reference()
                    
                elif pred in MODE_GESTURES:
                    new_stage = MODE_GESTURES[pred]
                    if gesture_stage != new_stage:
                        gesture_stage = new_stage
                        stable_label = pred
                        # Set 3D reference position when entering mode
                        wrist_tracker.set_reference(pos_3d)

            # USE 3D TRACKING FOR ACTIVE MODES
            if gesture_stage in ["changing_speed", "translating", "rotating"]:
                commands = wrist_tracker.get_movement_commands(pos_3d)
                displacement_info = wrist_tracker.get_displacement_info(pos_3d)
                
                # Filter commands based on current mode
                if gesture_stage == "changing_speed":
                    for cmd, magnitude in commands:
                        if cmd in ["ACCELERATE", "DECELERATE"]:
                            print(f"[3D GESTURE] {cmd} (dZ={magnitude:.1f}cm)")
                            stable_label = cmd
                            
                elif gesture_stage == "translating":
                    for cmd, magnitude in commands:
                        if cmd in ["MOVE LEFT", "MOVE RIGHT", "MOVE UP", "MOVE DOWN"]:
                            print(f"[3D GESTURE] {cmd} (d={magnitude:.1f}cm)")
                            stable_label = cmd
                            
                elif gesture_stage == "rotating":
                    for cmd, magnitude in commands:
                        if cmd in ["YAW LEFT", "YAW RIGHT"]:
                            print(f"[3D GESTURE] {cmd} (dX={magnitude:.1f}cm)")
                            stable_label = cmd
                
                # Display 3D displacement info
                if displacement_info:
                    disp_y = h - 120
                    cv2.putText(annotated_frame, f"3D Displacement:", 
                               (10, disp_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
                    cv2.putText(annotated_frame, f"  dX: {displacement_info['dx']:+.1f} cm", 
                               (10, disp_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    cv2.putText(annotated_frame, f"  dY: {displacement_info['dy']:+.1f} cm", 
                               (10, disp_y+35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    cv2.putText(annotated_frame, f"  dZ: {displacement_info['dz']:+.1f} cm", 
                               (10, disp_y+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                    cv2.putText(annotated_frame, f"  Dist: {displacement_info['distance']:.1f} cm", 
                               (10, disp_y+65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

            # Display 3D position
            cv2.putText(annotated_frame, f"3D: X={pos_3d[0]:.1f} Y={pos_3d[1]:.1f} Z={pos_3d[2]:.1f}cm", 
                       (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            
            # Display stage and active command
            cv2.putText(annotated_frame, f"Mode: {gesture_stage}", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if stable_label:
                cv2.putText(annotated_frame, f"Cmd: {stable_label}", 
                           (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Draw gesture label
            label_text = "unknown" if pred is None else f"{pred} ({conf:.2f})"
            annotated_frame = draw_gesture_label(annotated_frame, hand, label_text)

    # rgb->bgr for display/write
    output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    writer.write(output_frame)
    cv2.imshow('3D Hand Gesture Control', output_frame)

    # fps display
    fps_hist.append(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cv2.putText(output_frame, f"fps: {np.mean(fps_hist):.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    # keys
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('q'):
        print(f"[KEY] 'q' pressed")
   
        if knn_static.is_trained:
            knn_static.save("gesture_static.pkl")
        if knn_dynamic.is_trained:
            knn_dynamic.save("gesture_dynamic.pkl")

        print(f"[INFO] quitting program")
        break
    
    if k in STATIC_GESTURE_KEYS:
        current_label = STATIC_GESTURE_KEYS[k]
        print(f"[INFO] current label -> {current_label}")
    
    if k == ord('r'):
        recording = True
        samples_left = N_SAMPLES
        print(f"[INFO] recording {N_SAMPLES} frames for '{current_label}'")
    
    # Additional debug key to reset 3D filter
    if k == ord('f'):
        wrist_tracker.reset_filter()
        print("[INFO] 3D filter reset")

writer.release()
cap.release()
cv2.destroyAllWindows()
detector = None