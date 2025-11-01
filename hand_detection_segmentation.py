import cv2
import numpy as np
from collections import deque

### Segmentation based palm detection

# Configuration
wrist_extension_ratio = 1.2
min_area = 3000
max_area = 50000

# Initialize global objects
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=False
)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
prev_frames = deque(maxlen=3)

def detect_faces(frame):
    """Detect faces to exclude them."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def create_face_exclusion_mask(frame_shape, faces):
    """Create a mask that excludes face regions."""
    height, width = frame_shape[:2]
    face_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    for (fx, fy, fw, fh) in faces:
        padding = 30
        cv2.rectangle(face_mask, 
                     (max(0, fx - padding), max(0, fy - padding)),
                     (min(width, fx + fw + padding), min(height, fy + fh + padding)),
                     0, -1)
    
    return face_mask

def detect_skin(frame):
    """
    Robust skin segmentation combining YCrCb and HSV color spaces
    with adaptive thresholding.
    """
    # --- Convert color spaces ---
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Static ranges tuned for inclusivity ---
    ycrcb_lower = np.array([0, 133, 77], dtype=np.uint8)
    ycrcb_upper = np.array([255, 173, 127], dtype=np.uint8)

    hsv_lower = np.array([0, 15, 40], dtype=np.uint8)
    hsv_upper = np.array([25, 255, 255], dtype=np.uint8)

    mask_ycrcb = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)
    mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # --- Combine masks ---
    combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # --- Adaptive refinement ---
    # Compute global skin color bias (using the mask itself)
    skin_pixels = cv2.mean(ycrcb, mask=combined_mask)
    Cr_mean, Cb_mean = skin_pixels[1], skin_pixels[2]

    # Adjust thresholds around detected mean to adapt to lighting
    lower_adapt = np.array([0, max(0, Cr_mean - 20), max(0, Cb_mean - 20)], dtype=np.uint8)
    upper_adapt = np.array([255, min(255, Cr_mean + 20), min(255, Cb_mean + 20)], dtype=np.uint8)

    adapt_mask = cv2.inRange(ycrcb, lower_adapt, upper_adapt)
    final_mask = cv2.bitwise_and(combined_mask, adapt_mask)

    # --- Morphological cleanup ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return final_mask


def analyze_convexity_defects(contour):
    """
    Analyze convexity defects to find fingers
    Palms have defects between fingers, forearms don't
    """
    hull = cv2.convexHull(contour, returnPoints=False)
    
    if len(hull) <= 3 or len(contour) < 10:
        return [], 0, None
    
    try:
        defects = cv2.convexityDefects(contour, hull)
    except:
        return [], 0, None
    
    if defects is None:
        return [], 0, None
    
    # Analyze defects
    significant_defects = []
    
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        
        # Calculate depth and angle
        depth = d / 256.0
        
        # Calculate angle at far point
        a = np.linalg.norm(np.array(start) - np.array(far))
        b = np.linalg.norm(np.array(end) - np.array(far))
        c = np.linalg.norm(np.array(start) - np.array(end))
        
        if a == 0 or b == 0:
            continue
        
        angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b))
        angle_deg = np.degrees(angle)
        
        # Significant defects: deep enough and acute angle (between fingers)
        if depth > 20 and angle_deg < 90:
            significant_defects.append({
                'start': start,
                'end': end,
                'far': far,
                'depth': depth,
                'angle': angle_deg
            })
    
    return significant_defects, len(significant_defects), defects

def calculate_palm_score(contour):
    """
    Calculate a score indicating likelihood of being a palm
    Higher score = more likely to be palm (not forearm)
    """
    score = 0
    features = {}
    
    # Basic measurements
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    
    # 1. Aspect ratio - palms are more square, forearms are elongated
    aspect_ratio = float(w) / h if h > 0 else 0
    features['aspect_ratio'] = aspect_ratio
    
    if 0.6 < aspect_ratio < 1.7:
        score += 30  # Good palm aspect ratio
    elif aspect_ratio < 0.4 or aspect_ratio > 2.5:
        score -= 20  # Too elongated (likely forearm)
    
    # 2. Compactness - palms are more compact
    compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    features['compactness'] = compactness
    
    if compactness > 0.3:
        score += 20  # Compact shape
    
    # 3. Convexity defects (fingers!)
    defects, defect_count, _ = analyze_convexity_defects(contour)
    features['defect_count'] = defect_count
    
    if 2 <= defect_count <= 5:
        score += 40  # Typical number of finger gaps
    elif defect_count == 1:
        score += 20  # Partially closed hand
    elif defect_count == 0:
        score += 10  # Fist
    else:
        score -= 10  # Too many defects (noise)
    
    # 4. Solidity - ratio of contour area to convex hull area
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    features['solidity'] = solidity
    
    if solidity < 0.85:
        score += 20  # Concave (has fingers)
    else:
        score -= 10  # Too convex (maybe forearm)
    
    # 5. Circularity - palms are somewhat circular
    if area > 0:
        circularity = (perimeter ** 2) / (4 * np.pi * area)
        features['circularity'] = circularity
        
        if 1.2 < circularity < 4:
            score += 15  # Good palm-like circularity
    
    # 6. Bounding box area ratio
    bbox_area = w * h
    extent = float(area) / bbox_area if bbox_area > 0 else 0
    features['extent'] = extent
    
    if extent > 0.5:
        score += 10  # Fills bounding box well
    
    features['score'] = score
    return score, features, defects

def find_palm_center(contour, defects):
    """
    Find the approximate center of the palm
    Uses the centroid adjusted by finger positions
    """
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    # If we have finger defects, adjust center to be below them
    if len(defects) > 0:
        # Average y-position of far points (between fingers)
        avg_far_y = np.mean([d['far'][1] for d in defects])
        
        # Palm center is typically below the finger gaps
        # Adjust cy to be in the palm region
        if cy < avg_far_y:
            cy = int(avg_far_y + (cy - avg_far_y) * 0.3)
    
    return (cx, cy)

def extract_palm_region(contour, defects, wrist_ext_ratio):
    """
    Extract palm region based on finger gaps, extending to approximate wrist
    
    Args:
        contour: Hand contour
        defects: Detected finger gaps
        wrist_ext_ratio: How far below finger gaps to extend (1.0 = same height as gap width)
    
    Returns:
        palm_bbox: (x1, y1, x2, y2) for palm region
        wrist_point: Estimated wrist center point
    """
    x, y, w, h = cv2.boundingRect(contour)
    
    if len(defects) == 0:
        # No fingers detected - fallback to simple crop
        # Use upper 70% of detected region
        palm_h = int(h * 0.7)
        palm_bbox = (x, y, x + w, y + palm_h)
        wrist_point = (x + w // 2, y + palm_h)
        return palm_bbox, wrist_point
    
    # Get all defect (finger gap) points
    far_points = np.array([d['far'] for d in defects])
    
    # Find bounding region of finger gaps
    min_gap_x = np.min(far_points[:, 0])
    max_gap_x = np.max(far_points[:, 0])
    min_gap_y = np.min(far_points[:, 1])  # Topmost gap
    avg_gap_y = np.mean(far_points[:, 1])  # Average gap height
    
    # Calculate palm dimensions based on finger spread
    finger_spread_width = max_gap_x - min_gap_x
    
    # Palm width: slightly wider than finger spread (add 30% padding)
    palm_width = int(finger_spread_width * 1.3)
    
    # Center the palm bbox on the finger gaps
    palm_center_x = (min_gap_x + max_gap_x) // 2
    palm_x1 = max(0, palm_center_x - palm_width // 2)
    palm_x2 = palm_x1 + palm_width
    
    # Palm height: from top of gaps down to wrist
    # Typical palm is about 1.0-1.3x the width of finger spread
    palm_height = int(finger_spread_width * wrist_ext_ratio)
    
    # Start just above the highest gap (to include base of fingers)
    palm_y1 = max(0, min_gap_y - int(palm_height * 0.1))
    
    # Extend down to approximate wrist
    palm_y2 = palm_y1 + palm_height
    
    # Make sure we don't go outside the original contour bounds
    palm_y2 = min(palm_y2, y + h)
    palm_x2 = min(palm_x2, x + w)
    
    palm_bbox = (palm_x1, palm_y1, palm_x2, palm_y2)
    
    # Estimate wrist center point (bottom center of palm bbox)
    wrist_point = ((palm_x1 + palm_x2) // 2, palm_y2)
    
    return palm_bbox, wrist_point

def detect_palm_regions(frame):
    """
    Main detection function - finds palm regions only
    """
    # Detect faces
    faces = detect_faces(frame)
    face_mask = create_face_exclusion_mask(frame.shape, faces)
    
    # Get skin mask
    skin_mask = detect_skin(frame)
    
    # Apply face exclusion
    skin_mask = cv2.bitwise_and(skin_mask, face_mask)
    
    # Find contours
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Analyze each contour
    palm_candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Calculate palm likelihood score
        score, features, defects = calculate_palm_score(cnt)
        
        # Threshold score
        if score > 30:  # Minimum score to be considered a palm
            # Find palm center
            palm_center = find_palm_center(cnt, defects)
            
            # Extract palm region (excluding fingers)
            palm_bbox, wrist_point = extract_palm_region(cnt, defects, wrist_extension_ratio)
            
            palm_candidates.append({
                'contour': cnt,
                'score': score,
                'features': features,
                'defects': defects,
                'palm_bbox': palm_bbox,
                'palm_center': palm_center,
                'wrist_point': wrist_point,
                'full_bbox': cv2.boundingRect(cnt)
            })
    
    # Sort by score
    palm_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return palm_candidates, skin_mask, faces

def calculate_full_hand_bbox(contour, defects, palm_bbox, wrist_point):
    """
    Calculate full hand bounding box that includes fingertips and palm,
    but stops at the wrist level (no forearm).
    """

    palm_x1, palm_y1, palm_x2, palm_y2 = palm_bbox
    min_x, min_y = palm_x1, palm_y1
    max_x, max_y = palm_x2, palm_y2

    # Collect fingertip points from convexity defects (start/end of each defect)
    fingertip_points = []
    if len(defects) > 0:
        for defect in defects:
            fingertip_points.append(defect['start'])
            fingertip_points.append(defect['end'])
    else:
        # Fallback: use hull if no defects
        hull = cv2.convexHull(contour, returnPoints=True)
        fingertip_points = hull.reshape(-1, 2).tolist()

    if fingertip_points:
        fingertip_array = np.array(fingertip_points)
        tip_min_x = np.min(fingertip_array[:, 0])
        tip_max_x = np.max(fingertip_array[:, 0])
        tip_min_y = np.min(fingertip_array[:, 1])
        tip_max_y = np.max(fingertip_array[:, 1])

        # Extend palm bbox to include fingertips (top)
        min_x = min(min_x, tip_min_x)
        max_x = max(max_x, tip_max_x)
        min_y = min(min_y, tip_min_y)
        # But limit the bottom to the wrist line
        if wrist_point is not None:
            max_y = min(max_y, wrist_point[1] + 5)  # 5px margin just below wrist
        else:
            max_y = max(max_y, tip_max_y)

    # Add padding for visual comfort
    padding = 10
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = max_x + padding
    max_y = max_y + padding

    return (min_x, min_y, max_x, max_y)



def detect_palm_bboxes(frame, return_full_hand=False):
    """
    Detect palm bounding boxes
    
    Args:
        frame: Input frame
        return_full_hand: If True, returns full hand bbox including fingertips; 
                         if False, returns palm-only bbox
    
    Returns:
        List of bounding boxes, mask, faces, candidates
    """
    candidates, mask, faces = detect_palm_regions(frame)
    
    if return_full_hand:
        bboxes = []
        for c in candidates:
            full_bbox = calculate_full_hand_bbox(
                c['contour'], c['defects'], c['palm_bbox'], c['wrist_point']
            )
            bboxes.append(full_bbox)
            c['full_hand_bbox'] = full_bbox
    else:
        bboxes = [c['palm_bbox'] for c in candidates]

    return bboxes, mask, faces, candidates

def visualize_palm_detection(frame, candidates, mask=None, faces=None, show_analysis=True, show_full_hand=False):
    """Visualize palm detection with detailed analysis"""
    output = frame.copy()
    
    # Draw face exclusion zones
    if faces is not None:
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(output, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
            cv2.putText(output, "FACE", (fx, fy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw each palm candidate
    for i, candidate in enumerate(candidates):
        contour = candidate['contour']
        score = candidate['score']
        features = candidate['features']
        defects = candidate['defects']
        palm_bbox = candidate['palm_bbox']
        palm_center = candidate['palm_center']
        wrist_point = candidate['wrist_point']
        full_bbox = candidate['full_bbox']
        
        # Draw full hand contour (light green)
        cv2.drawContours(output, [contour], -1, (144, 238, 144), 1)
        
        # Choose which bbox to draw prominently
        if show_full_hand and 'full_hand_bbox' in candidate:
            # Draw full hand bbox (bright green, thick)
            x1, y1, x2, y2 = candidate['full_hand_bbox']
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            bbox_label = "Full Hand"
            
            # Draw palm bbox in background (light cyan, thin)
            px1, py1, px2, py2 = palm_bbox
            cv2.rectangle(output, (px1, py1), (px2, py2), (255, 255, 0), 1)
        else:
            # Draw palm-only bounding box (bright green, thick)
            x1, y1, x2, y2 = palm_bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            bbox_label = "Palm Only"
        
        # Draw wrist point and line
        if wrist_point:
            cv2.circle(output, wrist_point, 8, (255, 100, 0), -1)
            cv2.putText(output, "Wrist", (wrist_point[0] - 30, wrist_point[1] + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
            
            # Draw horizontal wrist line
            wx1 = palm_bbox[0]
            wx2 = palm_bbox[2]
            cv2.line(output, (wx1, wrist_point[1]), (wx2, wrist_point[1]), 
                    (255, 100, 0), 2)
        
        # Draw palm center
        if palm_center:
            cv2.circle(output, palm_center, 6, (255, 0, 255), -1)
        
        # Draw convexity defects (finger gaps) with emphasis
        if show_analysis and len(defects) > 0:
            # Get all gap points
            far_points = [d['far'] for d in defects]
            
            # Draw bounding box around finger gaps (visualization)
            if len(far_points) >= 2:
                gap_xs = [p[0] for p in far_points]
                gap_ys = [p[1] for p in far_points]
                min_gx, max_gx = min(gap_xs), max(gap_xs)
                min_gy, max_gy = min(gap_ys), max(gap_ys)
                
                # Draw finger gap region (cyan dashed)
                cv2.rectangle(output, (min_gx, min_gy), (max_gx, max_gy), 
                            (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(output, "Finger Gaps", (min_gx, min_gy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw each gap and fingertips
            for j, defect in enumerate(defects):
                start = defect['start']  # Fingertip
                end = defect['end']      # Fingertip
                far = defect['far']      # Gap point
                
                # Draw lines to gap
                cv2.line(output, start, far, (0, 255, 255), 1)
                cv2.line(output, end, far, (0, 255, 255), 1)
                
                # Mark gap point (cyan circle)
                cv2.circle(output, far, 7, (0, 255, 255), -1)
                cv2.circle(output, far, 10, (0, 200, 200), 2)
                
                # Mark fingertips (red circles) if showing full hand
                if show_full_hand:
                    cv2.circle(output, start, 5, (0, 0, 255), -1)
                    cv2.circle(output, end, 5, (0, 0, 255), -1)
                
                # Number the gaps
                cv2.putText(output, str(j+1), (far[0] + 12, far[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2)
        
        # Add information text
        hand_h = y2 - y1
        hand_w = x2 - x1
        info_y = y1 - 10
        texts = [
            f"{bbox_label} #{i+1} (Score: {score:.0f})",
            f"Fingers: {features['defect_count']} | Size: {hand_w}x{hand_h}px"
        ]
        
        for j, text in enumerate(texts):
            cv2.putText(output, text, (x1, info_y - j * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show mask in corner
    if mask is not None:
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_resized = cv2.resize(mask_color, (200, 150))
        output[10:160, 10:210] = mask_resized
        cv2.putText(output, "Skin Mask", (15, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return output


# Demo application
cap = cv2.VideoCapture(0)

print("=" * 60)
print("PALM-ONLY DETECTION (excludes forearms)")
print("=" * 60)
print("\nFeatures:")
print("  • Palm bbox calculated FROM finger gaps")
print("  • Width: 1.3x finger spread")
print("  • Height: 1.2x finger spread (down to wrist)")
print("  • Completely ignores forearm detection")
print("  • Face detection and exclusion")
print("\nControls:")
print("  • 'a' - Toggle analysis visualization")
print("  • 'f' - Toggle full hand / palm-only bbox")
print("  • '+' - Increase wrist extension")
print("  • '-' - Decrease wrist extension")
print("  • 'q' - Quit")
print("\nVisualization:")
print("  • Green box = Palm region (gap-based)")
print("  • Cyan circles = Finger gaps (anchor points)")
print("  • Orange circle = Wrist estimate")
print("  • Yellow box = Finger gap bounding region")
print("=" * 60)

show_analysis = True
show_full_hand = False
wrist_ratio = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # # Remove background here
    # frame = remove_background(frame, bg_color=(255, 255, 255), threshold=0.01)

    # Update wrist ratio
    wrist_extension_ratio = wrist_ratio

    # Detect palms
    bboxes, mask, faces, candidates = detect_palm_bboxes(frame, show_full_hand)

    # Visualize
    output = visualize_palm_detection(frame, candidates, mask, faces, show_analysis, show_full_hand)

    # Info text
    mode = "Full Hand" if show_full_hand else "Palm Only"
    info = f"Mode: {mode} | Palms: {len(candidates)} | Wrist Ext: {wrist_ratio:.1f}x"
    cv2.putText(output, info, (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Palm-Only Detection', output)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        show_analysis = not show_analysis
        print(f"Analysis visualization: {'ON' if show_analysis else 'OFF'}")
    elif key == ord('f'):
        show_full_hand = not show_full_hand
        print(f"Mode: {'Full Hand' if show_full_hand else 'Palm Only'}")
    elif key == ord('+') or key == ord('='):
        wrist_ratio += 0.1
        print(f"Wrist extension ratio: {wrist_ratio:.1f}x")
    elif key == ord('-') or key == ord('_'):
        wrist_ratio = max(0.5, wrist_ratio - 0.1)
        print(f"Wrist extension ratio: {wrist_ratio:.1f}x")

cap.release()
cv2.destroyAllWindows()
