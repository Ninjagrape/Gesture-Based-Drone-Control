"""
knn_hand_tracking.py - KNN model to recognise gestures from mediapipe handlandmarker 21 keypoints

Integrated with monocular depth tracking to provide drone control commands.
"""

# LIBRARIES----------------------------------------------------------------------------------------------------------------------------------------------
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
import time

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# IMPORT DEPTH TRACKING MODULE
from monocular_depth_tracking import Wrist3DTracker
from lp_filt import OneEuroFilter

#LANDMARK GLOBALS--------------------------------------------------------------------------------------
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


#KNN CLASSES------------------------------------------------------------------------------------
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
            print("[WARN] Not enough samples to train")
            return
        X = np.stack(self.X, axis=0)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, np.array(self.y))
        self.is_trained = True
        print(f"[INFO] Trained KNN: {len(self.y)} samples, classes={sorted(set(self.y))}")

    def predict(self, feats):
        if not self.is_trained:
            return None, 0.0
        fs = self.scaler.transform(feats.reshape(1, -1))
        pred = self.clf.predict(fs)[0]
        proba = float(self.clf.predict_proba(fs).max()) if hasattr(self.clf, "predict_proba") else 0.0
        return pred, proba
    
    def save(self, path="gesture_knn.pkl"):
        if not self.is_trained:
            print("[WARN] Model not trained, nothing saved")
            return
        payload = {"scaler": self.scaler, "clf": self.clf}
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Saved model → {path}")

    @classmethod
    def load(cls, path="gesture_knn.pkl"):
        if not os.path.isfile(path):
            print(f"[INFO] No saved model at {path}")
            return None
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj.scaler = payload["scaler"]
        obj.clf = payload["clf"]
        obj.is_trained = True
        print(f"[INFO] Loaded model ← {path}")
        return obj


class GestureHands:
    def __init__(self):
        self.landmark_history = {0: [], 1: []}

    def track_gesture_motion(self, hand_index, hand_landmarks):
        hist = self.landmark_history[hand_index]
        palm_center = hand_landmarks[0]
        hist.append([palm_center.x, palm_center.y, getattr(palm_center, "z", 0.0)])

        if len(hist) > 10:
            hist.pop(0)

        if len(hist) > 1:
            velocities = np.diff(np.asarray(hist, dtype=np.float32), axis=0)
            return np.mean(velocities, axis=0)
        return None


#DRAWING FUNCTIONS---------------------------------------------------------------------------
def draw_landmarks_on_image(image, detection_result):
    annotated_image = np.copy(image)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * annotated_image.shape[1])
                y = int(landmark.y * annotated_image.shape[0])
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

            connections = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20)
            ]
            for start_idx, end_idx in connections:
                s = hand_landmarks[start_idx]; e = hand_landmarks[end_idx]
                x1 = int(s.x * annotated_image.shape[1]); y1 = int(s.y * annotated_image.shape[0])
                x2 = int(e.x * annotated_image.shape[1]); y2 = int(e.y * annotated_image.shape[0])
                cv2.line(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated_image


def draw_gesture_label(image, hand_or_box, gesture, *, pad=12):
    if hand_or_box is None:
        return image
    h, w = image.shape[:2]

    xs = [int(lm.x * w) for lm in hand_or_box]
    ys = [int(lm.y * h) for lm in hand_or_box]
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
    cv2.putText(image, str(gesture), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return image


#HELPER FUNCTIONS-------------------------------------------------------------------------------------------
def palm_facing_camera(result):
    hand = result.handedness[0][0].category_name
    lm = result.hand_landmarks[0]

    v1 = np.array([lm[MCP[INDEX]].x - lm[0].x, lm[MCP[INDEX]].y - lm[0].y, 
                   getattr(lm[MCP[INDEX]], 'z', 0.0) - getattr(lm[0], 'z', 0.0)])
    v2 = np.array([lm[MCP[PINKY]].x - lm[0].x, lm[MCP[PINKY]].y - lm[0].y,
                   getattr(lm[MCP[PINKY]], 'z', 0.0) - getattr(lm[0], 'z', 0.0)])
    nz_palm = np.cross(v1,v2)[2]
    
    return (nz_palm <= 0) if hand == "Right" else (nz_palm >= 0)
        

def extract_features_from_hand(hand_landmarks):
    pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in hand_landmarks], dtype=np.float32)

    wrist = pts[WRIST].copy()
    rel = pts - wrist
    base_span = np.linalg.norm(pts[5] - pts[17]) or 1e-6
    rel /= base_span

    def angle(a, b, c):
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1) or 1e-6, np.linalg.norm(v2) or 1e-6
        return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

    angles = np.array([
        angle(rel[1], rel[2], rel[3]),   # thumb
        angle(rel[0], rel[5], rel[6]),   # index
        angle(rel[0], rel[9], rel[10]),  # middle
        angle(rel[0], rel[13], rel[14]), # ring
        angle(rel[0], rel[17], rel[18])  # pinky
    ], dtype=np.float32)

    return np.concatenate([rel.flatten(), angles], axis=0)


#MAIN SETUP----------------------------------------------------------------------------------------------------

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize 3D tracker from imported module
wrist_tracker = Wrist3DTracker(frame_width=w)

# MediaPipe setup - handle Windows path issues
model_path = 'hand_landmarker.task'
if os.path.isabs(model_path):
    model_path = os.path.basename(model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("annotated.mp4", fourcc, fps, (w, h))

# Controls
STATIC_GESTURE_KEYS = {
    ord('1'): "STOP", ord('2'): "HOLD", ord('3'): "CHANGE SPEED",
    ord('4'): "TRANSLATE", ord('5'): "ROTATE"
}

MODE_GESTURES = {
    "CHANGE SPEED": "changing_speed",
    "TRANSLATE": "translating",
    "ROTATE": "rotating",
}
EXIT_GESTURES = {"STOP", "HOLD"}

MIN_STATIC_CONF = 0.85
current_label = "STOP"
N_SAMPLES = 100
recording = False
samples_left = 0

# Load or create models
knn_static = KNNGesture.load("gesture_static.pkl") or KNNGesture(n_neighbors=7)
knn_dynamic = KNNGesture.load("gesture_dynamic.pkl") or KNNGesture(n_neighbors=7)

temporal_multi = GestureHands()
stable_label = None
gesture_stage = "idle"


filters = [
    OneEuroFilter(
        min_cutoff=1.0,
        beta=0.007 if i == 0 else 0.05,  # wrist (index 0) gets higher responsiveness
        d_cutoff=1.0
    )
    for i in range(21)
]

# Gesture recognition debouncer (per-hand)
gesture_hold_start = {}           # map: hand_index -> (candidate_label, start_time)
GESTURE_CONFIRM_TIME = 0.25        # seconds to confirm a mode gesture


print("\n" + "="*60)
print("3D GESTURE CONTROL SYSTEM (MODULAR)")
print("="*60)
print("\nCONTROLS:")
print("  1-5: Select gesture for training")
print("  R: Record 100 samples")
print("  F: Reset 3D filter")
print("  Q: Quit and save")
print("\nMODES:")
print("  STOP/HOLD     → Exit mode")
print("  CHANGE SPEED  → 3D depth control")
print("  TRANSLATE     → 3D translation")
print("  ROTATE        → 3D yaw control")
print("="*60 + "\n")

# Before the main loop, open a file for writing
output_file = open('thumb_tracking.txt', 'w')
output_file.write("# Thumb Tracking Data\n")
output_file.write("# thumb_tip_x, thumb_tip_y, thumb_tip_z\n")

# MAIN LOOP--------------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # Save thumb tip position (landmark 4)
    if detection_result.hand_world_landmarks:
        for hand_landmarks in detection_result.hand_world_landmarks:
            thumb_tip = hand_landmarks[0]  # Thumb tip is landmark index 4
            
            # Write to file: timestamp, x, y, z (in meters)
            output_file.write(f"{thumb_tip.x:.6f}, {thumb_tip.y:.6f}, {thumb_tip.z:.6f}\n")
            output_file.flush()  # Ensure data is written immediately


    # Filter all landmarks here
    if detection_result.hand_world_landmarks:
        for hand_landmarks in detection_result.hand_world_landmarks:
            timestamp = time.time()

            for i, landmark in enumerate(hand_landmarks):
                # Convert to numpy array
                pos = np.array([landmark.x, landmark.y, landmark.z])
                
                # Apply filter
                filtered_pos = filters[i].update(pos, timestamp)

                # Replace landmark values in-place
                landmark.x, landmark.y, landmark.z = filtered_pos

    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

    if detection_result.hand_landmarks:
        hands = list(detection_result.hand_landmarks)
        hands.sort(key=lambda hl: hl[WRIST].x)

        for hi, hand in enumerate(hands):
            # Get 3D position from depth module
            pos_3d = wrist_tracker.get_wrist_position_3d(hand, (h, w))
            
            # Temporal motion for gesture classification
            v = temporal_multi.track_gesture_motion(hi, hand)
            speed = np.linalg.norm(v) if v is not None else 0.0
            gesture_type = "dynamic" if speed > 0.02 else "static"

            # Extract features
            feats = extract_features_from_hand(hand)

            # === RECORDING MODE ===
            if recording and samples_left > 0:
                if gesture_type == "static":
                    knn_static.add_sample(feats, current_label)
                else:
                    motion_feats = v if v is not None else np.zeros(3)
                    combined = np.concatenate([feats, motion_feats])
                    knn_dynamic.add_sample(combined, current_label)

                samples_left -= 1
                cv2.putText(annotated_frame, f"Recording {current_label}: {N_SAMPLES - samples_left}/{N_SAMPLES}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                if samples_left == 0:
                    recording = False
                    knn_static.fit()
                    knn_dynamic.fit()
            
            # === PREDICTION ===
            pred, conf = ("none", 0.0)
            
            if gesture_type == "static" and knn_static.is_trained:
                pred, conf = knn_static.predict(feats)
                if pred in ("STOP", "HOLD") and not palm_facing_camera(detection_result):
                    pred, conf = "none", 0.0
                if conf < MIN_STATIC_CONF:
                    pred = "none"

            elif gesture_type == "dynamic" and knn_dynamic.is_trained:
                motion_feats = v if v is not None else np.zeros(3)
                combined = np.concatenate([feats, motion_feats])
                pred, conf = knn_dynamic.predict(combined)

            # === GESTURE STATE MACHINE ===
            # We keep a separate hold timer per hand index `hi` so multiple hands work correctly.
            if gesture_type == "static" and pred != "none" and conf >= MIN_STATIC_CONF:
                # Immediate exit gestures
                if pred in EXIT_GESTURES:
                    gesture_stage = "idle"
                    stable_label = pred
                    wrist_tracker.clear_reference()
                    # clear any pending hold for this hand
                    gesture_hold_start.pop(hi, None)

                # Mode gestures require confirmation (debounce)
                elif pred in MODE_GESTURES:
                    new_stage = MODE_GESTURES[pred]
                    entry = gesture_hold_start.get(hi)

                    # If different candidate or first time, start or restart timer
                    if entry is None or entry[0] != pred:
                        gesture_hold_start[hi] = (pred, time.time())
                        stable_label = pred  # immediate UI feedback (not yet confirmed)
                    else:
                        # same candidate, check elapsed time
                        _, t0 = entry
                        elapsed = time.time() - t0
                        if elapsed >= GESTURE_CONFIRM_TIME:
                            # Confirm the change only once
                            if gesture_stage != new_stage:
                                gesture_stage = new_stage
                                stable_label = pred
                                # Only set reference if we have a valid 3D position
                                if pos_3d is not None:
                                    wrist_tracker.set_reference(pos_3d)
                            # remove the hold timer so it won't re-trigger
                            gesture_hold_start.pop(hi, None)
            else:
                # If prediction disappears / low confidence / non-static => reset any pending hold for this hand
                gesture_hold_start.pop(hi, None)


            # === USE 3D TRACKING ===
            if gesture_stage in ["changing_speed", "translating", "rotating"]:
                commands = wrist_tracker.get_movement_commands(pos_3d)
                displacement_info = wrist_tracker.get_displacement_info(pos_3d)
                
                # Filter commands by mode
                for cmd, magnitude in commands:
                    show_cmd = False
                    if gesture_stage == "changing_speed" and cmd in ["ACCELERATE", "DECELERATE"]:
                        show_cmd = True
                    elif gesture_stage == "translating" and cmd in ["MOVE LEFT", "MOVE RIGHT", "MOVE UP", "MOVE DOWN"]:
                        show_cmd = True
                    elif gesture_stage == "rotating" and cmd in ["YAW LEFT", "YAW RIGHT"]:
                        show_cmd = True
                    
                    if show_cmd:
                        print(f"[CMD] {cmd} ({magnitude:.1f}cm)")
                        stable_label = cmd
                
                # Display displacement
                if displacement_info:
                    y_offset = h - 120
                    cv2.putText(annotated_frame, f"Displacement:", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    cv2.putText(annotated_frame, f"X:{displacement_info['dx']:+.1f} Y:{displacement_info['dy']:+.1f} Z:{displacement_info['dz']:+.1f}cm", 
                               (10, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            # Display info
            cv2.putText(annotated_frame, f"3D: X={pos_3d[0]:.1f} Y={pos_3d[1]:.1f} Z={pos_3d[2]:.1f}cm", 
                       (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(annotated_frame, f"Mode: {gesture_stage}", 
                       (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if stable_label:
                cv2.putText(annotated_frame, f"Cmd: {stable_label}", 
                           (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Draw gesture label
            label_text = f"{pred} ({conf:.2f})" if pred != "none" else "none"
            annotated_frame = draw_gesture_label(annotated_frame, hand, label_text)

    output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    writer.write(output_frame)
    cv2.imshow('3D Gesture Control', output_frame)

    # Keyboard
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('q'):
        if knn_static.is_trained:
            knn_static.save("gesture_static.pkl")
            print("[INFO] static gesture saved")
        if knn_dynamic.is_trained:
            knn_dynamic.save("gesture_dynamic.pkl")
            print("[INFO] dynamic gesture saved")

        print("[INFO] Quitting...")
        break
    
    if k in STATIC_GESTURE_KEYS:
        current_label = STATIC_GESTURE_KEYS[k]
        print(f"[INFO] Label → {current_label}")
    
    if k == ord('r'):
        recording = True
        samples_left = N_SAMPLES
        print(f"[INFO] Recording {N_SAMPLES} samples for '{current_label}'")
    
    if k == ord('f'):
        wrist_tracker.reset_filter()
        print("[INFO] Filter reset")

writer.release()
cap.release()
cv2.destroyAllWindows()
detector = None

output_file.close()
