import streamlit as st
import cv2
import numpy as np

# Streamlit App Setup
st.title("Traffic Density Estimator")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Constants
MIN_CONTOUR_AREA = 800
IOU_THRESHOLD_NMS = 0.3
MAX_MISSED_FRAMES = 10

# Initialize variables in session state
if 'prev_frame' not in st.session_state:
    st.session_state.prev_frame = None
if 'persistent_detections' not in st.session_state:
    st.session_state.persistent_detections = []

# Helper Functions
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

def non_max_suppression_fast(boxes, overlap_thresh):
    if len(boxes) == 0:
        return []
    boxes_arr = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes], dtype=float)
    pick = []
    x1, y1, x2, y2 = boxes_arr[:, 0], boxes_arr[:, 1], boxes_arr[:, 2], boxes_arr[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [len(idxs) - 1]
        for pos in range(len(idxs) - 1):
            i = idxs[pos]
            xx1, yy1 = max(x1[last], x1[i]), max(y1[last], y1[i])
            xx2, yy2 = min(x2[last], x2[i]), min(y2[last], y2[i])
            w_, h_ = max(0, xx2 - xx1 + 1), max(0, yy2 - yy1 + 1)
            overlap = float(w_ * h_) / area[i]
            if overlap > overlap_thresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return [boxes[i] for i in pick]

def update_persistent_detections(current_boxes):
    updated = []
    matched_current = [False] * len(current_boxes)

    for detection in st.session_state.persistent_detections:
        best_iou = 0
        best_idx = -1
        for idx, curr_box in enumerate(current_boxes):
            iou = compute_iou(detection['box'], curr_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou > 0.3:
            detection['box'] = current_boxes[best_idx]
            detection['age'] = 0
            matched_current[best_idx] = True
            updated.append(detection)
        else:
            detection['age'] += 1
            if detection['age'] <= MAX_MISSED_FRAMES:
                updated.append(detection)

    for idx, flag in enumerate(matched_current):
        if not flag:
            updated.append({'box': current_boxes[idx], 'age': 0})
    
    st.session_state.persistent_detections = updated

# Video Processing
if video_file:
    tfile = open("temp_video.mp4", 'wb')
    tfile.write(video_file.read())
    cap = cv2.VideoCapture("temp_video.mp4")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    frame_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        fg_mask = bg_subtractor.apply(frame, learningRate=0.005)

        # Frame differencing
        if st.session_state.prev_frame is not None:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_prev = cv2.cvtColor(st.session_state.prev_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray_frame, gray_prev)
            _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.bitwise_or(fg_mask, diff_mask)

        st.session_state.prev_frame = frame.copy()

        # Morphological ops
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_boxes = [(x, y, w, h) for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA
                         for (x, y, w, h) in [cv2.boundingRect(cnt)]]

        current_boxes = non_max_suppression_fast(current_boxes, IOU_THRESHOLD_NMS)
        update_persistent_detections(current_boxes)

        final_boxes = [d['box'] for d in st.session_state.persistent_detections]

        for (x, y, w, h) in final_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        vehicle_area = sum([w * h for (x, y, w, h) in final_boxes])
        road_area = frame.shape[0] * frame.shape[1]
        density_ratio = vehicle_area / road_area

        if density_ratio < 0.2:
            density, color = "LOW", (0, 255, 0)
        elif density_ratio < 0.5:
            density, color = "MEDIUM", (0, 255, 255)
        else:
            density, color = "HIGH", (0, 0, 255)

        cv2.putText(frame, f"Detected vehicles: {len(final_boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Traffic density: {density}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Convert BGR to RGB and display
        frame_display.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
