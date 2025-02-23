import os
import cv2
import json
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ====================================================
# Pose-based Head Estimation (Improved Pose-Back Method)
# ====================================================
mp_pose = mp.solutions.pose

def compute_pose_head_info(image_bgr, min_confidence=0.5, offset_factor=0.7):
    """
    Uses MediaPipe Pose to estimate the head center for a back-view image.
    Returns a dictionary containing:
      - refined_head: estimated head center (x, y)
      - shoulder_width: distance between shoulders
      - image_width: width of the image
    Returns None if detection fails.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=min_confidence) as pose:
        results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    h, w, _ = image_bgr.shape
    if len(landmarks) < 13:
        return None

    left_shoulder = (landmarks[11].x * w, landmarks[11].y * h)
    right_shoulder = (landmarks[12].x * w, landmarks[12].y * h)
    shoulder_center = ((left_shoulder[0] + right_shoulder[0]) / 2,
                       (left_shoulder[1] + right_shoulder[1]) / 2)
    shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
    estimated_head = (shoulder_center[0], shoulder_center[1] - offset_factor * shoulder_width)
    nose = (landmarks[0].x * w, landmarks[0].y * h)
    if nose[1] < shoulder_center[1]:
        refined_head = ((estimated_head[0] + nose[0]) / 2, (estimated_head[1] + nose[1]) / 2)
    else:
        refined_head = estimated_head
    # Require that the head center is at least 0.1 * shoulder_width above the shoulders
    if shoulder_center[1] - refined_head[1] < 0.1 * shoulder_width:
        return None
    return {"refined_head": refined_head, "shoulder_width": shoulder_width, "image_width": w}

def compute_pose_metric(keyframes):
    """
    Computes the maximum normalized displacement (normalized by Address image width)
    between the refined head centers of the keyframes "Address", "Top", and "Impact".
    Returns the maximum normalized displacement (a float) or None if any detection fails.
    """
    pose_results = {}
    for key in ["Address", "Top", "Impact"]:
        info = compute_pose_head_info(keyframes[key])
        if info is None:
            return None
        pose_results[key] = info["refined_head"]
    # Compute pairwise Euclidean distances
    cA = np.array(pose_results["Address"])
    cT = np.array(pose_results["Top"])
    cI = np.array(pose_results["Impact"])
    d_AT = np.linalg.norm(cT - cA)
    d_TI = np.linalg.norm(cI - cT)
    d_AI = np.linalg.norm(cI - cA)
    max_disp = max(d_AT, d_TI, d_AI)
    global_width = keyframes["Address"].shape[1]
    return max_disp / global_width

# ====================================================
# YOLOv8 Head Detection Metrics
# ====================================================
def detect_head_yolov8(image_bgr, yolo_model, conf_threshold=0.1):
    """
    Runs YOLOv8 head detection with a customizable confidence threshold.
    Returns a tuple (bbox, conf) where bbox is [x1, y1, x2, y2] and conf is the detection confidence.
    Returns None if no detection meets the threshold.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None
    
    best_box = None
    best_conf = 0.0
    
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > best_conf and conf >= conf_threshold:
            best_conf = conf
            best_box = box
            
    if best_box is None:
        return None
    
    bbox = best_box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    return bbox, best_conf

def compute_iou(bbox1, bbox2):
    """
    Computes Intersection-over-Union (IoU) between two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    area1 = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
    union_area = area1 + area2 - inter_area
    return 0 if union_area == 0 else inter_area / union_area

def compute_yolo_metrics(keyframes, yolo_model, conf_threshold=0.1):
    """
    Runs YOLO detection on "Address", "Top", "Impact".
    If at least two detections exist, computes:
      - The normalized displacement between centers (normalized by avg bbox width)
      - The IoU between detected bounding boxes.
    Returns (max_disp_yolo, avg_iou_yolo), or (None, None) if not enough detections.
    """
    detections = {}
    for key in ["Address", "Top", "Impact"]:
        det = detect_head_yolov8(keyframes[key], yolo_model, conf_threshold)
        if det is not None:
            bbox, conf = det
            center = np.array(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
            width = bbox[2] - bbox[0]
            detections[key] = {"bbox": bbox, "center": center, "width": width}
    
    if len(detections) < 2:
        return None, None  # Keep None for missing YOLO metrics
    
    disp_list = []
    iou_list = []
    keys_det = list(detections.keys())
    
    for i in range(len(keys_det)):
        for j in range(i+1, len(keys_det)):
            k1, k2 = keys_det[i], keys_det[j]
            disp = np.linalg.norm(detections[k2]["center"] - detections[k1]["center"])
            avg_width = (detections[k1]["width"] + detections[k2]["width"]) / 2.0
            norm_disp = disp / avg_width if avg_width > 0 else 0
            disp_list.append(norm_disp)
            iou_list.append(compute_iou(detections[k1]["bbox"], detections[k2]["bbox"]))
    
    max_disp_yolo = max(disp_list) if disp_list else None
    avg_iou_yolo = max(iou_list) if iou_list else None
    
    return max_disp_yolo, avg_iou_yolo

# ====================================================
# Optical Flow for Head Region
# ====================================================
def compute_optical_flow_head(prev_img, curr_img, pose_info):
    """
    Computes sparse optical flow (using Lucas–Kanade) within an ROI defined around the head.
    The ROI is centered on the refined head (from pose_info) with side-length equal to the shoulder width.
    Returns the maximum displacement (in pixels) within the ROI.
    """
    refined_head = pose_info["refined_head"]
    shoulder_width = pose_info["shoulder_width"]
    cx, cy = refined_head
    half_size = shoulder_width / 2
    x1 = int(max(cx - half_size, 0))
    y1 = int(max(cy - half_size, 0))
    x2 = int(min(cx + half_size, prev_img.shape[1]))
    y2 = int(min(cy + half_size, prev_img.shape[0]))
    
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    roi_prev = prev_gray[y1:y2, x1:x2]
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_pts = cv2.goodFeaturesToTrack(roi_prev, mask=None, **feature_params)
    if prev_pts is None or len(prev_pts) == 0:
        return 0
    prev_pts += np.array([[x1, y1]], dtype=np.float32)
    
    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    valid_prev = prev_pts[status.flatten() == 1]
    valid_curr = curr_pts[status.flatten() == 1]
    if len(valid_prev) == 0:
        return 0
    displacements = np.linalg.norm(valid_curr - valid_prev, axis=1)
    return np.max(displacements)

# ====================================================
# Utility: Load Keyframes from Folder
# ====================================================
def load_keyframes(folder_path):
    """
    Expects keyframes named "Address.jpg", "Top.jpg", and "Impact.jpg" in the folder.
    Returns a dictionary mapping these keys to loaded BGR images.
    """
    keys = ["Address", "Top", "Impact"]
    keyframes = {}
    for key in keys:
        path = os.path.join(folder_path, f"{key}.jpg")
        if not os.path.exists(path):
            print(f"[{folder_path}] Missing {key}.jpg")
            return None
        img = cv2.imread(path)
        if img is None:
            print(f"[{folder_path}] Could not load {key}.jpg")
            return None
        keyframes[key] = img
    return keyframes

# ====================================================
# Compute Metrics for a Sequence
# ====================================================
def compute_sequence_metrics(folder_path, yolo_model):
    """
    For a given folder (with keyframes "Address", "Top", "Impact"),
    compute the following metrics:
      - pose_max_disp_norm: Maximum normalized displacement (pose-based) between refined head centers.
      - yolo_max_disp_norm: Maximum normalized displacement (YOLO) between detected centers (normalized by average bbox width).
      - yolo_avg_iou: Average IoU among the detected YOLO boxes.
      - optical_flow_max_disp_norm: Maximum optical flow displacement (within a head ROI) normalized by Address image width.
    Returns a dictionary of metrics.
    """
    keyframes = load_keyframes(folder_path)
    if keyframes is None:
        print(f"[{folder_path}] Could not load keyframes for metrics computation.")
        return None
    global_width = keyframes["Address"].shape[1]

    # Pose metric
    pose_metric = compute_pose_metric(keyframes)

    # YOLO metrics
    yolo_disp, yolo_iou = compute_yolo_metrics(keyframes, yolo_model)

    # Optical Flow metric
    pose_info_addr = compute_pose_head_info(keyframes["Address"])
    pose_info_top  = compute_pose_head_info(keyframes["Top"])
    if pose_info_addr is None or pose_info_top is None:
        flow_metric = None
    else:
        flow_AT = compute_optical_flow_head(keyframes["Address"], keyframes["Top"], pose_info_addr)
        flow_TI = compute_optical_flow_head(keyframes["Top"], keyframes["Impact"], pose_info_top)
        if flow_AT is not None and flow_TI is not None:
            flow_metric = max(flow_AT, flow_TI) / global_width
        else:
            flow_metric = None

    return {
        "folder": os.path.basename(folder_path),
        "pose_max_disp_norm": pose_metric,
        "yolo_max_disp_norm": yolo_disp,
        "yolo_avg_iou": yolo_iou,
        "optical_flow_max_disp_norm": flow_metric
    }

# ====================================================
# Custom JSON Encoder to handle NumPy types
# ====================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if obj is None:
            return None
        return super(NumpyEncoder, self).default(obj)

# ====================================================
# Main Driver: Process Folders and Save Metrics
# ====================================================
def main():
    top_level_dir = "event"  # Directory with keyframe subfolders
    if not os.path.isdir(top_level_dir):
        print(f"Top-level directory '{top_level_dir}' does not exist!")
        return
    subfolders = [f for f in os.listdir(top_level_dir) if os.path.isdir(os.path.join(top_level_dir, f))]
    subfolders.sort()

    # Load YOLOv8 head detection model (ensure "yolov8n-head.pt" is available)
    yolo_model = YOLO("yolov8n-head.pt")

    all_metrics = []
    for sub in subfolders:
        folder_path = os.path.join(top_level_dir, sub)
        print(f"\nProcessing folder: {folder_path}")
        metrics = compute_sequence_metrics(folder_path, yolo_model)
        if metrics is not None:
            all_metrics.append(metrics)
            print(f"[{sub}] Metrics: {metrics}")
        else:
            print(f"Processing failed for {folder_path}")

    # Save the metrics to a JSON file using the custom encoder
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, cls=NumpyEncoder)
    print("\nSaved metrics for all sequences to 'metrics.json'.")

if __name__ == "__main__":
    main()