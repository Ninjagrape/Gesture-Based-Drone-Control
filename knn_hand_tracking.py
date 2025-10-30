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



from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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

# # knn that auto-clamps k to the fitted sample count
# class KNNGesture:
#     def __init__(self, n_neighbors=5):
#         self.base_k = n_neighbors  # remember requested k
#         self.scaler = StandardScaler()
#         self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
#         self.X, self.y = [], []
#         self.is_trained = False

#     def _effective_k(self):
#         # determine k <= number of fitted samples (and >=1)
#         nfit = len(getattr(self.clf, "_fit_X", []))
#         return max(1, min(self.base_k, nfit)) if nfit else 1

#     def _apply_k(self):
#         # update the classifier's k right before use
#         k_eff = self._effective_k()
#         if getattr(self.clf, "n_neighbors", None) != k_eff:
#             self.clf.n_neighbors = k_eff  # safe to tweak between calls

#     def add_sample(self, feats, label):
#         self.X.append(feats.astype(np.float32))
#         self.y.append(label)

#     def fit(self):
#         if len(self.X) < 2:
#             print("[WARN] not enough samples to train")
#             return
#         X = np.stack(self.X, axis=0)
#         Xs = self.scaler.fit_transform(X)
#         self.clf.fit(Xs, np.array(self.y))
#         self.is_trained = True
#         self._apply_k()  # clamp k to len(fit)
#         print(f"[INFO] trained knn on {len(self.y)} samples, classes={sorted(set(self.y))}")

#     def predict(self, feats):
#         if not self.is_trained or not hasattr(self.clf, "_fit_X") or len(self.clf._fit_X) == 0:
#             return "none", 0.0
#         self._apply_k()  # re-clamp in case you’ve added more later
#         fs = self.scaler.transform(feats.reshape(1, -1))
#         pred = self.clf.predict(fs)[0]
#         proba = float(self.clf.predict_proba(fs).max()) if hasattr(self.clf, "predict_proba") else 0.0
#         return pred, proba




#CLASSES------------------------------------------------------------------------------------
class KNNGesture:
    def __init__(self, n_neighbors=5):
        self.base_k = n_neighbors  # remember requested k
        self.scaler = StandardScaler()
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        self.X, self.y = [], []
        self.is_trained = False
        
    def _effective_k(self):
        # determine k <= number of fitted samples (and >=1)
        nfit = len(getattr(self.clf, "_fit_X", []))
        return max(1, min(self.base_k, nfit)) if nfit else 1

    def _apply_k(self):
        # update the classifier's k right before use
        k_eff = self._effective_k()
        if getattr(self.clf, "n_neighbors", None) != k_eff:
            self.clf.n_neighbors = k_eff  # safe to tweak between calls
            

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
        self._apply_k()  # clamp k to len(fit)
        print(f"[INFO] trained knn on {len(self.y)} samples, classes={sorted(set(self.y))}")

    def predict(self, feats):
        if not self.is_trained or not hasattr(self.clf, "_fit_X") or len(self.clf._fit_X) == 0:
            return "none", 0.0
        self._apply_k()  # re-clamp in case you’ve added more later
        fs = self.scaler.transform(feats.reshape(1, -1))
        pred = self.clf.predict(fs)[0]
        proba = float(self.clf.predict_proba(fs).max()) if hasattr(self.clf, "predict_proba") else 0.0
        return pred, proba


    # def predict(self, feats):
    #     if not self.is_trained:
    #         return None, 0.0
    #     fs = self.scaler.transform(feats.reshape(1, -1))
    #     pred = self.clf.predict(fs)[0]
    #     proba = float(self.clf.predict_proba(fs).max()) if hasattr(self.clf, "predict_proba") else 0.0
    #     return pred, proba
    
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

    def process_multiple_hands(self, multi_hand_landmarks):
        """process gestures involving multiple hands"""
        if len(multi_hand_landmarks) < 2:
            return None

        # extract features from both hands using your existing extractor
        features = []
        for hand_landmarks in multi_hand_landmarks:
            hand_features = extract_features_from_hand(hand_landmarks)  # uses your function
            features.extend(hand_features.tolist())

        # relative positions between hands (wrist→wrist)
        h1 = multi_hand_landmarks[0][0]
        h2 = multi_hand_landmarks[1][0]
        relative_position = [h2.x - h1.x, h2.y - h1.y, getattr(h2, "z", 0.0) - getattr(h1, "z", 0.0)]
        features.extend(relative_position)

        return np.asarray(features, dtype=np.float32)



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
def get_hand_landmark(detection_result, print_statement="no"):
    """ returns hand landmark x,y,z information if hand is detected in frame """
    if detection_result.hand_landmarks:
        first = detection_result.hand_landmarks[0]
        landmark = [{"x": float(lm.x), "y": float(lm.y), "z": float(getattr(lm, "z", 0.0))} for lm in first]
        if print_statement == "yes":
            print(landmark)
        return landmark

def get_relative_distance_hand_to_camera(hand_landmarks):
    """ estimate relative hand–camera distance using hand size in image space. """
    if not hand_landmarks:
        return None
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    width  = max(xs) - min(xs)
    height = max(ys) - min(ys)
    hand_area = width * height
    relative_distance = 1.0 / max(hand_area, 1e-6)
    return relative_distance

def palm_facing_camera(result):
    hand = result.handedness[0][0].category_name  # "Right" or "Left"

    lm = get_hand_landmark(result)

    v1 = np.array([lm[MCP[INDEX]][k] - lm[0][k] for k in ('x','y','z')])
    v2 = np.array([lm[MCP[PINKY]][k] - lm[0][k] for k in ('x','y','z')])
    nz_palm = np.cross(v1,v2)[2]
    
    #the reason these are backwards is because camera is flipped when streaming
    if hand == "Right":
        if nz_palm > 0: 
            # print("L showing back hand")
            return False
        else: 
            # print(" L palm facing camera")
            return True
    
    if hand == "Left":
        if nz_palm < 0: 
            # print("R showing back hand")
            return False
        else: 
            # print("R palm facing camera")
            return True
        
def check_which_hand(result):
    hand = result.handedness[0][0].category_name  # "Right" or "Left"
    
    #the reason these are backwards is because camera is flipped when streaming
    if hand == "Right":
        print("LEFT HAND")
        return "Left"
    
    if hand == "Left":
        print("RIGHT HAND")
        return "Right"
    
        


# use this version in place of your current extract_features_from_hand
def extract_features_from_hand(hand_landmarks):
    """ build a compact, scale/translation-invariant feature vector. """
    # build wrist-centred, scale-normalised points
    pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in hand_landmarks], dtype=np.float32)
    rel = pts - pts[WRIST]
    base_span = np.linalg.norm(pts[5] - pts[17]) or 1e-6
    rel /= base_span

    # a few base angles (as before)
    def angle(a, b, c):
        v1, v2 = a - b, c - b
        n1 = np.linalg.norm(v1) or 1e-6
        n2 = np.linalg.norm(v2) or 1e-6
        cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
        return np.arccos(cosang)

    thumb_ang = angle(rel[1],  rel[2],  rel[3])
    idx_ang   = angle(rel[0],  rel[5],  rel[6])
    mid_ang   = angle(rel[0],  rel[9],  rel[10])
    ring_ang  = angle(rel[0],  rel[13], rel[14])
    pinky_ang = angle(rel[0],  rel[17], rel[18])
    angles = np.array([thumb_ang, idx_ang, mid_ang, ring_ang, pinky_ang], dtype=np.float32)

    # new: per-finger bits + count
    ext_bits = finger_extended_mask(rel).astype(np.float32)
    ext_count = np.array([ext_bits.sum() / 5.0], dtype=np.float32)

    # 63 rel coords + 5 angles + 5 bits + 1 count
    feats = np.concatenate([rel.flatten(), angles, ext_bits, ext_count], axis=0)
    return feats


# minimal: per-finger extended bits using your global indices
# rel: (21,3) already wrist-centred & scale-normalised
def finger_extended_mask(rel):
    # thumb uses cmc→mcp vs cmc→tip; others use mcp→pip vs mcp→tip
    chains = [
        (CMC[0],        MCP[THUMB],  FINGERTIPS[THUMB]),   # thumb:  cmc, mcp, tip
        (MCP[INDEX],    PIP[0],      FINGERTIPS[INDEX]),   # index:  mcp, pip, tip
        (MCP[MIDDLE],   PIP[1],      FINGERTIPS[MIDDLE]),  # middle: mcp, pip, tip
        (MCP[RING],     PIP[2],      FINGERTIPS[RING]),    # ring:   mcp, pip, tip
        (MCP[PINKY],    PIP[3],      FINGERTIPS[PINKY]),   # pinky:  mcp, pip, tip
    ]
    mask = np.zeros(5, dtype=bool)

    # thresholds to tune on your data
    min_tip_mcp  = 0.35   # tip–mcp span (in base_span units)
    max_bend_deg = 50.0   # smaller = straighter finger

    for i, (a, b, t) in enumerate(chains):
        v1 = rel[t] - rel[a]                      # direction to tip
        v2 = rel[b] - rel[a]                      # direction to next joint
        n1 = np.linalg.norm(v1) or 1e-6
        n2 = np.linalg.norm(v2) or 1e-6
        cosang = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
        bend = np.degrees(np.arccos(cosang))      # bend angle at 'a'
        tip_mcp = np.linalg.norm(rel[t] - rel[a]) # straightness proxy

        # finger is 'extended' if long and straight enough
        mask[i] = (tip_mcp >= min_tip_mcp) and (bend <= max_bend_deg)

    return mask


def kn_ready(kn):
    return kn and kn.is_trained and hasattr(kn.clf, "_fit_X") and len(kn.clf._fit_X) >= 1


#MAIN()??----------------------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("annotated.mp4", fourcc, fps, (w, h))

# recording + controls
GESTURE_KEYS = {
    ord('1'): "STOP",
    ord('2'): "GO",
    ord('3'): "HOLD",
    ord('4'): "CHANGE SPEED",
    ord('5'): "TRANSLATE",
    ord('6'): "ROTATE",
 

}

MIN_STATIC_CONF = 0.85  # static prediction confidence to accept a mode switch
XY_DEAD = 0.01          # deadzone for vx, vy
Z_DEAD  = 0.01          # deadzone for vz

# static gestures that switch/clear modes
MODE_GESTURES = {
    "STOP":         "stopped", 
    "CHANGE SPEED": "changing_speed",
    "TRANSLATE":    "translating",
    "ROTATE":       "rotating"

}
EXIT_GESTURES = {"STOP", "HOLD"}  # these drop back to idle


current_label = "STOP"
N_SAMPLES = 100
recording = False
samples_left = 0

frame_idx = 0



knn_static  = KNNGesture.load("gesture_static.pkl")  or KNNGesture(n_neighbors=5)
knn_dynamic = KNNGesture.load("gesture_dynamic.pkl") or KNNGesture(n_neighbors=5)

fps_hist = deque(maxlen=30)

temporal_multi = GestureHands()

#used for dynamic gestures 
stable_label = None
gesture_stage = "idle"  # idle -> ready -> dynamic


print("\n")
print("-------------------------------------------------------")
print("HOW TO RUN:")
print("Press KEY 1-6 to set label")
print("KEY label legend: 1: STOP, 2: GO, 3: HOLD, 4: CHANGE SPEED, 5: TRANSLATE, 6: ROTATE")
print("Press KEY['r'] to record samples")
print("Press KEY['q'] to quit and save trained model/s")
print("\n")

print("If you have a trained model, it will automatically run the system \n on this trained model the next time it is run.")
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

    # detect hands (tasks api)
    detection_result = detector.detect(mp_image)

    # base overlay (skeleton)
    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

    if detection_result.hand_landmarks:
        # stabilise order: left→right by wrist x
        hands = list(detection_result.hand_landmarks)
        hands.sort(key=lambda hl: hl[WRIST].x)

        # two_hand_feats = temporal_multi.process_multiple_hands(hands)
        
        # wrist_3d, _, _ = get_wrist_position_3d(output_frame)
        


        for hi, hand in enumerate(hands):
            # temporal motion (vx, vy, vz) for dynamic gestures
            v = temporal_multi.track_gesture_motion(hi, hand)  # None or array(3,)
            if v is not None:
                vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
            else:
                vx, vy, vz = 0.0, 0.0, 0.0
                
            # classify motion magnitude
            speed = np.linalg.norm(v) if v is not None else 0.0
            
            
            gesture_type = "dynamic" if speed > 0.005 else "static"
            
            # if v is not None:
            #     if gesture_type == "dynamic":
            #         print(f"[DEBUG] dynamic | vx={v[0]:+.3f}, vy={v[1]:+.3f}, vz={v[2]:+.3f}, speed={speed:.3f}")
            #     else:
            #         print(f"[DEBUG] static  | vx={v[0]:+.3f}, vy={v[1]:+.3f}, vz={v[2]:+.3f}, speed={speed:.3f}")
            # else:
            #     print("[DEBUG] no motion history yet")


            # your existing path
            rel_dist = get_relative_distance_hand_to_camera(hand)
            feats = extract_features_from_hand(hand)
            
            # rebuild rel for the quick guard 
            pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0)] for lm in hand], dtype=np.float32)
            rel = pts - pts[WRIST]
            rel /= (np.linalg.norm(pts[5] - pts[17]) or 1e-6)
            ext_bits = finger_extended_mask(rel)
            ext_cnt  = int(ext_bits.sum())

            
            if recording and samples_left > 0:
                motion_feats = v if v is not None else np.zeros(3)
                speed = np.linalg.norm(v) if v is not None else 0.0
                gesture_type = "dynamic" if speed > 0.015 else "static"

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
            
            pred, conf = ("none", 0.0)
            speed = np.linalg.norm(v) if v is not None else 0.0
            gesture_type = "dynamic" if speed > 0.02 else "static"
            
            # if ord("0"):
            #     print(f"[DEBUG] {ext_cnt} fingers extended")
            if gesture_type == "static" and knn_static.is_trained:
                pred, conf = knn_static.predict(feats)

                # optional palm check just for stop/hold
                if pred in ("STOP", "GO") and not palm_facing_camera(detection_result):
                    pred, conf = "none", 0.0

                # if you want to forbid mapping partials to "STOP"
                if 1 <= ext_cnt <= 4 and pred == "STOP":
                    pred = "none"  # or map to a "partial" class you define
                    print(f"[DEBUG] 5 fingers not extended, detected {ext_cnt}, predicting {pred}")
                    
                # if (1 <= ext_cnt or ext_cnt >2)  and pred == "TRANSLATE":
                #     pred = "none"
                # #     print(f"[DEBUG] 2 fingers not extended, detected {ext_cnt}, predicting {pred}")
                
                # if (ext_cnt >2)  and pred == "ROTATE":
                #     pred = "none"
                    
                # if ext_cnt != 2 and pred == "GO":
                #     pred = "none"
                #     print(f"[DEBUG] 2 fingers not extended, detected {ext_cnt}, predicting {pred}")
                
                # if ext_cnt !=2 and pred == "ROTATE":
                #     pred = "none"
                #     print(f"[DEBUG] 2 fingers not extended, detected {ext_cnt}, predicting {pred}")

                if conf < 0.90:
                    pred = "none"

            
            elif gesture_type == "dynamic" and kn_ready(knn_dynamic):
                motion_feats = v if v is not None else np.zeros(3, dtype=np.float32)
                combined = np.concatenate([feats, motion_feats])
                pred, conf = knn_dynamic.predict(combined)

            #DEALING WITH THE DYNAMIC GESTURES 
            if gesture_type == "static" and pred != "none" and conf >= MIN_STATIC_CONF:
                # exit to idle on stop/hold
                if pred in EXIT_GESTURES:
                    gesture_stage = "idle"
                    stable_label  = pred  # e.g., STOP/HOLD/GO 
                # enter one of the sticky modes
                elif pred in MODE_GESTURES:
                    new_stage = MODE_GESTURES[pred]
                    if gesture_stage != new_stage:
                        gesture_stage = new_stage
                        stable_label  = pred  # remember last static trigger (optional)
        
        

            # --- per-frame behaviour while in a sticky mode (runs regardless of static/dynamic) ---
            if gesture_stage == "stopped":
                print("[INFO] Drone stopped. GO gesture required to continue.")
                if not (pred == "GO" and conf >= MIN_STATIC_CONF):
                    pred, conf = "none", 0.0
                else:
                    gesture_stage = "idle"           # resume normal operation
                    stable_label  = "GO"
                print("[STATE] GO recognised. resuming from STOPPED → idle.")
                
            elif gesture_stage == "changing_speed":
                # use hand z-velocity: negative = towards camera
                #CHANG LOGIC FOR THIS 
                if v is not None:
                    if v[2] < -Z_DEAD: 
                        print("[GESTURE] ACCELERATE")
                        stable_label = "ACCELERATE"
                    elif v[2] > +Z_DEAD:
                        print("[GESTURE] SLOW_DOWN")
                        stable_label = "SLOW_DOWN"
                        
            # #speed based
            # elif gesture_stage == "translating":
            #     # map x/y velocity to planar moves (multiple can fire if you want diagonals)
            #     if v is not None:
            #         if v[0] < -XY_DEAD:
            #             print("[GESTURE] MOVE LEFT")
            #             stable_label = "MOVE LEFT"
            #         elif v[0] > +XY_DEAD:
            #             print("[GESTURE] MOVE RIGHT")
            #             stable_label = "MOVE RIGHT"
            #         if v[1] < -XY_DEAD:
            #             print("[GESTURE] MOVE UP")
            #             stable_label = "MOVE UP"
            #         elif v[1] > +XY_DEAD:
            #             print("[GESTURE] MOVE DOWN")
            #             stable_label = "MOVE DOWN"
                        
            
            elif gesture_stage == "translating":
                # infer direction from where the fingers are pointing, not from motion
                if hand is not None:
                    # use index + middle fingertips as primary pointing direction
                    tip_idx = [FINGERTIPS[INDEX], FINGERTIPS[MIDDLE]]
                    mcp_idx = [MCP[INDEX], MCP[MIDDLE]]
                    # midpoint of both fingers
                    tip_mid = np.mean([rel[t] for t in tip_idx], axis=0)
                    mcp_mid = np.mean([rel[m] for m in mcp_idx], axis=0)

                    # pointing vector (normalised)
                    dir_vec = tip_mid - mcp_mid
                    mag = np.linalg.norm(dir_vec)
                    if mag < 1e-6:
                        continue
                    dir_vec /= mag

                    # interpret direction relative to camera axes
                    dx, dy = dir_vec[0], dir_vec[1]

                    # thresholds — tune 0.25–0.35 range depending on jitter
                    DIR_T = 0.3

                    if abs(dx) > abs(dy):
                        if dx > DIR_T:
                            print("[GESTURE] MOVE RIGHT")
                            stable_label = "MOVE RIGHT"
                        elif dx < -DIR_T:
                            print("[GESTURE] MOVE LEFT")
                            stable_label = "MOVE LEFT"
                    else:
                        if dy < -DIR_T and palm_facing_camera(detection_result):
                            print("[GESTURE] MOVE UP")
                            stable_label = "MOVE UP"
                        elif dy > DIR_T:
                            print("[GESTURE] MOVE DOWN")
                            stable_label = "MOVE DOWN"
                            
                            
            elif gesture_stage == "rotating":
                                
                # thumb_tip = hand[FINGERTIPS[THUMB]]
                # pinky_tip = hand[FINGERTIPS[PINKY]]
                # index_tip = hand[FINGERTIPS[INDEX]]
                # wrist     = hand[WRIST]

                # # approximate 3D vectors (convert from normalised image coords to physical cm)
                # def norm_to_3d(lm):
                #     return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

                # thumb = norm_to_3d(thumb_tip)
                # pinky = norm_to_3d(pinky_tip)
                # index = norm_to_3d(index_tip)
                # wrist = norm_to_3d(wrist)

                # # build axes
                # x_axis = thumb - pinky               # across palm
                # y_axis = index - wrist               # roughly palm “height”
                # z_axis = np.cross(x_axis, y_axis)    # palm normal
                
                
                # # normalise
                # x_axis /= (np.linalg.norm(x_axis) or 1e-6)
                # y_axis /= (np.linalg.norm(y_axis) or 1e-6)
                # z_axis /= (np.linalg.norm(z_axis) or 1e-6)

                # # euler-like angles (in degrees)
                # yaw   = np.degrees(np.arctan2(z_axis[0], z_axis[2]))      # left–right rotation
                # pitch = np.degrees(np.arctan2(-z_axis[1], z_axis[2]))     # tilt toward/away camera
                # roll  = np.degrees(np.arctan2(x_axis[1], x_axis[0]))      # twist around axis
                
                # print(f"yaw: {yaw}")
                # print(f"pitch: {pitch}")
                # print(f"roll: {roll}")
                
                
                # handedness = check_which_hand(detection_result)
                
                # if handedness == "Left":
                #     yaw, roll = -yaw, -roll
                    
                        
                # ANG_T = 20.0

                # if yaw > ANG_T:
                #     print("[GESTURE] YAW RIGHT")
                # elif yaw < -ANG_T:
                #     print("[GESTURE] YAW LEFT")

                # if pitch > ANG_T:
                #     print("[GESTURE] PITCH DOWN")
                # elif pitch < -ANG_T:
                #     print("[GESTURE] PITCH UP")

                # if roll > ANG_T:
                #     print("[GESTURE] ROLL CW" if handedness == "Left" else "[GESTURE] ROLL CCW")
                # elif roll < -ANG_T:
                #     print("[GESTURE] ROLL CCW" if handedness == "Left" else "[GESTURE] ROLL CW")
                stable_label = "rotating.."
                pass 


                                                

                    
            
            cv2.putText(annotated_frame, f"stage: {gesture_stage}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)


            # simple dynamic overlay (you can map velocity to move labels if desired)
            label_text = "unknown" if pred is None else f"{pred} ({conf:.3f})"
            annotated_frame = draw_gesture_label(annotated_frame, hand, label_text)
            cv2.putText(annotated_frame, f"v=({vx:+.3f},{vy:+.3f})", (10, 72 + 18*hi),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
            if rel_dist is not None:
                cv2.putText(annotated_frame, f"distance~{rel_dist:.2f}", (10, 52 + 18*hi),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)


    # rgb->bgr for display/write
    output_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    writer.write(output_frame)
    cv2.imshow('Hand Pose Detection (KNN)', output_frame)

    # fps display (simple smoothing)
    fps_hist.append(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cv2.putText(output_frame, f"fps: {np.mean(fps_hist):.1f}", (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    # keys
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('1'):
        print("[KEY] '1' pressed")
    elif k == ord('2'):
        print("[KEY] '2' pressed")
    elif k == ord('3'):
        print("[KEY] '3' pressed")
    elif k == ord('4'):
        print("[KEY] '4' pressed")
    elif k == ord('5'):
        print("[KEY] '5' pressed")
    elif k == ord('6'):
        print("[KEY] '6' pressed")
    elif k == ord('7'):
        print("[KEY] '7' pressed")
    elif k == ord('8'):
        print("[KEY] '8' pressed")
    elif k == ord('9'):
        print("[KEY] '9' pressed")
    
    if k == ord('q'):
        print(f"[KEY] 'q' pressed")
   
        if knn_static.is_trained:
            knn_static.save("gesture_static.pkl")
        if knn_dynamic.is_trained:
            knn_dynamic.save("gesture_dynamic.pkl")

        print(f"[INFO] quitting program")
        break
    if k in GESTURE_KEYS:
        current_label = GESTURE_KEYS[k]
        print(f"[INFO] current label -> {current_label}")
    if k == ord('r'):
        recording = True
        samples_left = N_SAMPLES
        print(f"[INFO] recording {N_SAMPLES} frames for '{current_label}'")
    

        

writer.release()
cap.release()
cv2.destroyAllWindows()
detector = None
