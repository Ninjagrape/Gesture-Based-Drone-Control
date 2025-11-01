import cv2
import numpy as np
import mediapipe as mp

# Configuration
MIN_CONTOUR_AREA = 3000       # ignore tiny contours
MAX_CONTOUR_AREA_FR = 0.5    # ignore contours covering >50% of frame
SKIN_KERNEL_SIZE = 5          # morphological kernel for cleaning
ROI_MIN_SIZE = 120            # smallest accepted palm box side
ROI_TARGET_SIZE = 256         # MediaPipe input size
MAX_HANDS = 2                 # detect both hands
SMOOTHING_ALPHA = 0.45        # exponential smoothing factor for box tracking
ASSOC_IOU_THRESHOLD = 0.30    # IoU threshold to link tracks
LINE_THICKNESS = 2            # drawing line width

# Functions
# Convert bgr to rgb
def convert_bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Compute IoU between two rectangle box for 2 hand tracking
def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)

# Face mask with cascade
def detect_faces_haar(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.2, 5)

# Create black rectangle mask to ignore face region
def create_face_mask(frame, faces):
    h, w = frame.shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    for (x, y, fw, fh) in faces:
        # Small buffer around the face
        pad = int(0.25 * fh)  
        cv2.rectangle(mask,
                      (max(0, x - pad), max(0, y - pad)),
                      (min(w, x + fw + pad), min(h, y + fh + pad)),
                      0, -1)
    return mask

# Skin segmentation HSV + YCrCb
def generate_skin_mask(frame):

    # Detect skin with segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Skin tones ranges
    mask_hsv1 = cv2.inRange(hsv, (0, 25, 40), (25, 255, 255))
    mask_hsv2 = cv2.inRange(hsv, (160, 25, 40), (180, 255, 255))
    mask_ycc  = cv2.inRange(ycc, (0, 128, 70), (255, 180, 140))

    # Combine HSV + YCrCb results
    combined_mask = cv2.bitwise_and(cv2.bitwise_or(mask_hsv1, mask_hsv2), mask_ycc)

    # Tighten mask if enough pixels detected
    if cv2.countNonZero(combined_mask) > 50:
        mean_ycc = cv2.mean(ycc, mask=combined_mask)
        Cr, Cb = mean_ycc[1], mean_ycc[2]
        lo = np.array([0, max(0, Cr - 22), max(0, Cb - 22)], np.uint8)
        hi = np.array([255, min(255, Cr + 22), min(255, Cb + 22)], np.uint8)
        combined_mask = cv2.bitwise_and(combined_mask, cv2.inRange(ycc, lo, hi))

    # Morphological smoothing to remove small holes and speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SKIN_KERNEL_SIZE, SKIN_KERNEL_SIZE))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined_mask

# Palm contour detection 
def detect_rotated_palm_box(contour, skin_mask):
    x, y, w, h = cv2.boundingRect(contour)
    projection = np.sum(skin_mask[y:y+h, x:x+w] > 0, axis=1)
    if len(projection) < 5:
        return None
    projection_smooth = cv2.GaussianBlur(projection.astype(np.float32).reshape(-1, 1), (1, 9), 0).ravel()
    diff = np.diff(projection_smooth)
    drop_index = np.argmin(diff)  
    wrist_y = int(y + drop_index)
    # Fallback if drop is weak
    if projection_smooth[drop_index] > 0.8 * projection_smooth.max():
        wrist_y = int(y + 0.8 * h)
    # Keep contour line above wrist
    trimmed_points = np.array([pt for pt in contour[:, 0, :] if pt[1] < wrist_y], np.int32)
    if len(trimmed_points) < 5:
        trimmed_points = contour.reshape(-1, 2)
    rect = cv2.minAreaRect(trimmed_points)
    (cx, cy), (rw, rh), angle = rect
    # Normalize so width ≥ height
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0

    # Expand region for better detection
    rw *= 1.1
    rh *= 1.05
    return (cx, cy, rw, rh, angle)

# Extract coordinates 
def rect_to_box_points(cx, cy, w, h, angle):
    rect = ((cx, cy), (w, h), angle)
    poly = cv2.boxPoints(rect).astype(int)
    x1, y1 = poly[:, 0].min(), poly[:, 1].min()
    x2, y2 = poly[:, 0].max(), poly[:, 1].max()
    return (x1, y1, x2, y2), poly

# ROI for mediapipe to process
# Extract the palm and make sure crops upright 
def extract_upright_palm_roi(frame, rect):
    cx, cy, w, h, angle = rect
    rotation_matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, rotation_matrix, (frame.shape[1], frame.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(rotated.shape[1], x2), min(rotated.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None, None, None, None
    roi = rotated[y1:y2, x1:x2]
    inverse_matrix = cv2.invertAffineTransform(rotation_matrix)
    return roi, inverse_matrix, (x1, y1), (x1, y1, x2, y2)

# Map mediapipe landmark coordinates to video frame
def map_landmarks_to_frame(landmarks, Minv, crop_origin, roi_size):
    rw, rh = roi_size
    ox, oy = crop_origin
    points = []
    for lm in landmarks.landmark:
        xr, yr = lm.x * rw + ox, lm.y * rh + oy
        xo = Minv[0, 0] * xr + Minv[0, 1] * yr + Minv[0, 2]
        yo = Minv[1, 0] * xr + Minv[1, 1] * yr + Minv[1, 2]
        points.append((int(xo), int(yo)))
    return points

# Palm  track
# Convert palm countour box to axis bounding box for IoU
def rect_to_axb(rect):
    cx, cy, w, h, angle = rect
    (x1, y1, x2, y2), _ = rect_to_box_points(cx, cy, w, h, angle)
    return (x1, y1, x2, y2)

# Smooth bounding box between frames
def smooth_rectangles(old, new, alpha):
    ocx, ocy, ow, oh, oa = old
    ncx, ncy, nw, nh, na = new
    da = ((na - oa + 180) % 360) - 180  
    return (ocx + alpha * (ncx - ocx),
            ocy + alpha * (ncy - ocy),
            ow + alpha * (nw - ow),
            oh + alpha * (nh - oh),
            oa + alpha * da)

# Palm container
class PalmTrack:
    def __init__(self, rect):
        self.rect = rect
        self.age = 0

# Track up to two hands 
class PalmTracker:
    def __init__(self, alpha=SMOOTHING_ALPHA, iou_thresh=ASSOC_IOU_THRESHOLD, max_tracks=MAX_HANDS):
        self.alpha, self.iou_thresh, self.max_tracks = alpha, iou_thresh, max_tracks
        self.tracks = []
    # Associate ti existing dection with smoothing
    def update(self, rects):
        det_boxes = [rect_to_axb(r) for r in rects]
        used = set()
        # Update existing tracks
        for t in self.tracks:
            best_idx, best_iou = -1, 0
            for i, box in enumerate(det_boxes):
                if i in used: continue
                score = compute_iou(rect_to_axb(t.rect), box)
                if score > best_iou:
                    best_iou, best_idx = score, i
            if best_idx >= 0 and best_iou >= self.iou_thresh:
                t.rect = smooth_rectangles(t.rect, rects[best_idx], self.alpha)
                used.add(best_idx)
                t.age = min(t.age + 1, 10)
            else:
                t.age -= 1
        # Create new tracks
        for i, r in enumerate(rects):
            if i not in used and len(self.tracks) < self.max_tracks:
                self.tracks.append(PalmTrack(r))
        # Remove stale tracks
        self.tracks = [t for t in self.tracks if t.age > -3]
        return [t.rect for t in self.tracks]

# Dection function
# Coombine all mask to identify palm region
def detect_palm_regions(frame, face_cascade):
    faces = detect_faces_haar(frame, face_cascade)
    mask_skin = generate_skin_mask(frame)
    mask_faces = create_face_mask(frame, faces)
    combined_mask = cv2.bitwise_and(mask_skin, mask_faces)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = combined_mask.shape
    palm_rects = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA_FR * H * W:
            continue
        rect = detect_rotated_palm_box(contour, combined_mask)
        if rect is None:
            continue
        axb, _ = rect_to_box_points(*rect)
        if (axb[2] - axb[0]) < ROI_MIN_SIZE or (axb[3] - axb[1]) < ROI_MIN_SIZE:
            continue
        palm_rects.append(rect)

    return palm_rects[:MAX_HANDS], combined_mask, faces

# Mediappipe hand landmark
mp_hands = mp.solutions.hands

# Feed ROI to mediapipe 
def run_mediapipe_hands(hands_detector, roi_bgr):
    roi_rgb = convert_bgr_to_rgb(cv2.resize(roi_bgr, (ROI_TARGET_SIZE, ROI_TARGET_SIZE)))
    return hands_detector.process(roi_rgb)
# Draw 21 handland marks
def draw_hand_connections(frame, landmarks):
    PALM_CONN = [(0,1),(0,5),(5,9),(9,13),(13,17),(0,17)]
    FINGER_CONN = [(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
                   (9,10),(10,11),(11,12),(13,14),(14,15),
                   (15,16),(17,18),(18,19),(19,20)]
    for a, b in PALM_CONN + FINGER_CONN:
        if a < len(landmarks) and b < len(landmarks):
            cv2.line(frame, landmarks[a], landmarks[b], (40, 220, 255), LINE_THICKNESS)

# Main live camera
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not working")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    tracker = PalmTracker(alpha=SMOOTHING_ALPHA)

    print("=" * 70)
    print("Hand detection")
    print("=" * 70)

    show_debug = True
    # persistence buffer
    last_rects = []  

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=MAX_HANDS,
                        model_complexity=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as mp_detector:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            # Palm Detection
            rects_raw, mask, faces = detect_palm_regions(frame, face_cascade)
            rects_smooth = tracker.update(rects_raw)

            if rects_smooth:
                last_rects = rects_smooth
            elif len(last_rects) > 0:
                rects_smooth = last_rects[-2:]
            else:
                rects_smooth = []

            # Skin mask
            if show_debug:
                for (x, y, fw, fh) in faces:
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
                sm = cv2.cvtColor(cv2.resize(mask, (200, 150)), cv2.COLOR_GRAY2BGR)
                frame[10:160, 10:210] = sm
                cv2.putText(frame, "Skin Mask", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Hand Detection
            any_detected = False
            if rects_smooth:
                for rect in rects_smooth:
                    cx, cy, w, h, ang = rect
                    roi, Minv, crop_origin, _ = extract_upright_palm_roi(frame, rect)
                    if roi is None:
                        continue

                    result = run_mediapipe_hands(mp_detector, roi)
                    if result.multi_hand_landmarks:
                        any_detected = True
                        # Palm dection with mediapipe landmark
                        axb, poly = rect_to_box_points(*rect)
                        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
                        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 255), -1)

                        for hand_lms in result.multi_hand_landmarks:
                            pts = map_landmarks_to_frame(
                                hand_lms, Minv, crop_origin, (roi.shape[1], roi.shape[0])
                            )
                            for p in pts:
                                cv2.circle(frame, p, 3, (0, 200, 255), -1)
                            draw_hand_connections(frame, pts)
                    else:
                        # Fallback to mediapipe if no palm detected
                        fallback_result = mp_detector.process(convert_bgr_to_rgb(frame))
                        if fallback_result.multi_hand_landmarks:
                            any_detected = True
                            for hand_lms in fallback_result.multi_hand_landmarks:
                                # Compute bounding box from landmarks
                                xs = [lm.x * W for lm in hand_lms.landmark]
                                ys = [lm.y * H for lm in hand_lms.landmark]
                                x1, y1 = int(min(xs)), int(min(ys))
                                x2, y2 = int(max(xs)), int(max(ys))
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                for p in zip(xs, ys):
                                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 200, 255), -1)
                                draw_hand_connections(frame, [(int(x), int(y)) for x, y in zip(xs, ys)])

            # Fallback to mediapipe to feed whole frame to mediapipe
            if not any_detected:
                fallback_result = mp_detector.process(convert_bgr_to_rgb(frame))
                if fallback_result.multi_hand_landmarks:
                    for hand_lms in fallback_result.multi_hand_landmarks:
                        any_detected = True
                        xs = [lm.x * W for lm in hand_lms.landmark]
                        ys = [lm.y * H for lm in hand_lms.landmark]
                        x1, y1 = int(min(xs)), int(min(ys))
                        x2, y2 = int(max(xs)), int(max(ys))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        for p in zip(xs, ys):
                            cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 200, 255), -1)
                        draw_hand_connections(frame, [(int(x), int(y)) for x, y in zip(xs, ys)])

            # Display info
            #fps = cap.get(cv2.CAP_PROP_FPS)
            #cv2.putText(frame, f"Hands: {len(rects_smooth)}  FPS: {fps:.1f}",
            #    (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Hand tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# 
if __name__ == "__main__":
    main()