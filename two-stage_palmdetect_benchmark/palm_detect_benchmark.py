# Palm_detect_benchmark is used to evaluate the accuracy of the traditional palm detection module against the Google Mediapipe 
# model as ground truth. The traditional palm detection model will have green colour output whereas 
# Google Mediapipe will have yellow output. It will also return a classification report at the end with an analysis
# of the accuracy for both "Hand" detected and "No Hand" detected categories

# Insert this under main loop of two_stage_detection.py to set up the benchmark




from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def benchmark_live():
    print("=" * 70)
    print(" Palm detection vs Baseline MediaPipe")
    print(" Press 'q' to quit")
    print("=" * 70)

    # Initialise camera setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    tracker = PalmTracker(alpha=0.25)
    mp_hands_module = mp.solutions.hands
    drawer = mp.solutions.drawing_utils

    # Initialise benchmark variables
    y_true, y_pred = [], []
    total_frames = 0
    detected_custom = 0
    detected_baseline = 0
    t0 = time.time()

    # Memory for palm tracking and frame smoothing
    last_roi = None
    lm_history = []
    smooth_pts = None
    miss_count = 0
    last_valid_landmarks = []
    last_seen = time.time()

    # Mediapipe line model and two-stage palm detection model
    with mp_hands_module.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.65
    ) as baseline_mp, mp_hands_module.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=1,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.65
    ) as hybrid_mp:

        try:
            while True:
                # Read and mirror webcam frame
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                total_frames += 1
                H, W = frame.shape[:2]

                # Use Meiapipe as ground truth 
                base_result = baseline_mp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                base_detected = base_result.multi_hand_landmarks is not None
                # Visualise Google Mediapipe hand landmark model in yellow
                if base_detected:
                    for hand in base_result.multi_hand_landmarks:
                        drawer.draw_landmarks(
                            frame, hand, mp_hands_module.HAND_CONNECTIONS,
                            drawer.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2),
                            drawer.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2)
                        )
                        pts = np.array([(int(lm.x * W), int(lm.y * H)) for lm in hand.landmark])
                        x1, y1 = np.min(pts, axis=0)
                        x2, y2 = np.max(pts, axis=0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Two-stage palm detection model
                rects_raw, mask, _ = detect_palm_regions(frame, face_cascade)
                rects_smooth = tracker.update(rects_raw)
                custom_detected = False
                palm_landmarks = []
                fullframe_landmarks = []
                palm_boxes = []
                mediapipe_boxes = []

                # Extract upright ROI to feed Mediapipe 
                for rect in rects_smooth:
                    cx, cy, w, h, ang = rect
                    roi, Minv, crop_origin, _ = extract_upright_palm_roi(frame, rect)
                    if roi is None:
                        roi = last_roi
                    else:
                        last_roi = roi
                    if roi is None:
                        continue
                    roi_area = w * h
                    aspect_ratio = w / (h + 1e-6)
                    if roi_area < 25000 or 0.6 < aspect_ratio < 1.4:
                        continue
                    
                    # Run Mediapiepe with feeded ROI
                    result = run_mediapipe_hands(hybrid_mp, roi)
                    if result.multi_hand_landmarks:
                        custom_detected = True
                        _, poly = rect_to_box_points(*rect)
                        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
                        cv2.circle(frame, (int(cx), int(cy)), 4, (255, 0, 255), -1)
                        palm_boxes.append(poly)

                        for hand_lms in result.multi_hand_landmarks:
                            pts = map_landmarks_to_frame(
                                hand_lms, Minv, crop_origin, (roi.shape[1], roi.shape[0])
                            )
                            # 2-frame bufferfor smoothing
                            lm_history.append(np.array(pts, dtype=np.float32))
                            if len(lm_history) > 2:
                                lm_history.pop(0)
                            avg_pts = np.mean(lm_history, axis=0).astype(np.float32)
                            if smooth_pts is None:
                                smooth_pts = avg_pts
                            smooth_pts = 0.6 * smooth_pts + 0.4 * avg_pts
                            pts = [tuple(p.astype(int)) for p in smooth_pts]
                            palm_landmarks.append(pts)
                            # Visulaise hand landmark
                            for p in pts:
                                cv2.circle(frame, p, 3, (0, 255, 0), -1)
                            draw_hand_connections(frame, pts)

                # Fallback to Mediapipe if nothing detected
                # Set a counter
                if palm_landmarks:
                    miss_count = 0
                else:
                    miss_count += 1
                # When miss count is greater than 2 then fall back to Mediapipe with full frame
                if miss_count > 2:
                    fallback_result = hybrid_mp.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if fallback_result.multi_hand_landmarks:
                        custom_detected = True
                        for hand_lms in fallback_result.multi_hand_landmarks:
                            xs = [lm.x * W for lm in hand_lms.landmark]
                            ys = [lm.y * H for lm in hand_lms.landmark]
                            pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
                            fullframe_landmarks.append(pts)
                            x1, y1 = int(min(xs)), int(min(ys))
                            x2, y2 = int(max(xs)), int(max(ys))
                            mediapipe_boxes.append((x1, y1, x2, y2))

                # Merge both detections
                final_landmarks = palm_landmarks or fullframe_landmarks
                final_boxes = palm_boxes or mediapipe_boxes

                # Presistance buffer for hand disappearance
                if len(final_landmarks) > 0:
                    last_valid_landmarks = (final_landmarks, 2)
                    last_seen = time.time()
                else:
                    elapsed = time.time() - last_seen
                    if last_valid_landmarks:
                        old_lms, ttl = last_valid_landmarks
                        if elapsed < 0.05 and ttl > 0:
                            final_landmarks = old_lms
                            last_valid_landmarks = (old_lms, ttl - 1)
                        else:
                            last_valid_landmarks = []

                # Draw two-stage palm detection result in green 
                for pts in final_landmarks:
                    for p in pts:
                        cv2.circle(frame, p, 3, (0, 255, 0), -1)
                    draw_hand_connections(frame, pts)
                for box in final_boxes:
                    if isinstance(box, np.ndarray):
                        cv2.polylines(frame, [box], True, (0, 255, 0), 2)
                    else:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Set counter for each model
                if custom_detected:
                    detected_custom += 1
                if base_detected:
                    detected_baseline += 1

                y_true.append(1 if base_detected else 0)
                y_pred.append(1 if custom_detected else 0)

                # Display result
                elapsed = time.time() - t0
                fps = total_frames / elapsed if elapsed > 0 else 0
                # Calculate the total frames 
                cv2.putText(frame, f"Frames: {total_frames} | FPS: {fps:.1f}",
                            (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Google mediapipe shown in yellow and two-stage detection model in green colour
                cv2.putText(frame, f"Google mediapipe model: {'YES' if base_detected else 'NO'}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255) if base_detected else (0, 0, 255), 2)
                cv2.putText(frame, f"Palm detection model:   {'YES' if custom_detected else 'NO'}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if custom_detected else (0, 0, 255), 2)

                cv2.imshow("Two-stage model Benchmark", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nExiting\n")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Benchmark summary
            if len(y_true) > 0:
                print("\n\n BENCHMARK SUMMARY ")
                try:
                    cm = confusion_matrix(y_true, y_pred)
                    report = classification_report(
                        y_true, y_pred, target_names=["No Hand", "Hand"], digits=3
                    )
                    acc = accuracy_score(y_true, y_pred)
                    print("Confusion Matrix:\n", cm)
                    print("\nClassification Report:\n", report)
                    print("------------------------------------")
                except Exception as e:
                    print(" Error ", e)
