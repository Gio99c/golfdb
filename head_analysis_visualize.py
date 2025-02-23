import os
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# ====================================================
# Pose-based Head Estimation (Improved Pose-Back Method)
# ====================================================
mp_pose = mp.solutions.pose

def compute_pose_head_info(image_bgr, min_confidence=0.5, offset_factor=0.7):
    """
    Uses MediaPipe Pose to compute key landmarks and estimate the head center for back-view images.
    Returns a dictionary with:
      - left_shoulder (landmark 11)
      - right_shoulder (landmark 12)
      - shoulder_center (average of shoulders)
      - shoulder_width (distance between shoulders)
      - refined_head: an estimated head center.
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
    if shoulder_center[1] - refined_head[1] < 0.1 * shoulder_width:
        return None
    return {
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "shoulder_center": shoulder_center,
        "shoulder_width": shoulder_width,
        "nose": nose,
        "refined_head": refined_head
    }

def visualize_pose_overlay(image_bgr, pose_info):
    """
    Draws circles for key landmarks and the refined head center.
    """
    vis = image_bgr.copy()
    cv2.circle(vis, (int(pose_info["left_shoulder"][0]), int(pose_info["left_shoulder"][1])), 5, (255, 0, 0), -1)
    cv2.circle(vis, (int(pose_info["right_shoulder"][0]), int(pose_info["right_shoulder"][1])), 5, (255, 0, 0), -1)
    cv2.circle(vis, (int(pose_info["shoulder_center"][0]), int(pose_info["shoulder_center"][1])), 5, (0, 255, 0), -1)
    cv2.circle(vis, (int(pose_info["nose"][0]), int(pose_info["nose"][1])), 5, (0, 0, 255), -1)
    cv2.circle(vis, (int(pose_info["refined_head"][0]), int(pose_info["refined_head"][1])), 7, (0, 255, 255), -1)
    cv2.putText(vis, "Head", (int(pose_info["refined_head"][0]), int(pose_info["refined_head"][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return vis

# ====================================================
# YOLOv8 Head Detection
# ====================================================
def detect_head_yolov8(image_bgr, yolo_model):
    """
    Runs YOLOv8 head detection on the given BGR image.
    Returns (bbox, conf) where bbox is [x1, y1, x2, y2] and conf is the detection confidence.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(image_rgb)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None
    best_box = None
    best_conf = 0.0
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best_box = box
    if best_box is None:
        return None
    bbox = best_box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
    return bbox, best_conf

def overlay_yolo(image, bbox, conf):
    """
    Draws a bounding box and the detection confidence on the image.
    """
    vis = image.copy()
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"Head: {conf:.2f}", (x1, max(y1-10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return vis

# ====================================================
# Optical Flow for Head Region
# ====================================================
def compute_optical_flow_head(prev_img, curr_img, pose_info):
    """
    Computes sparse optical flow (using Lucas-Kanade) in the head region,
    defined by a ROI based on the refined head coordinate and shoulder width from pose_info.
    
    Returns a visualization image (current frame with flow arrows drawn) and
    the average displacement vector (dx, dy).
    """
    refined_head = pose_info["refined_head"]
    shoulder_width = pose_info["shoulder_width"]
    cx, cy = refined_head
    half_size = shoulder_width / 2
    # Define ROI around the head
    x1 = int(max(cx - half_size, 0))
    y1 = int(max(cy - half_size, 0))
    x2 = int(min(cx + half_size, prev_img.shape[1]))
    y2 = int(min(cy + half_size, prev_img.shape[0]))
    
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    
    # Crop ROI
    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]
    
    # Detect features in the previous ROI
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    prev_pts = cv2.goodFeaturesToTrack(prev_roi, mask=None, **feature_params)
    if prev_pts is None:
        return curr_img.copy(), (0, 0)
    # Adjust points to full image coordinates
    prev_pts += np.array([[x1, y1]], dtype=np.float32)
    
    # Compute optical flow using Lucas-Kanade
    lk_params = dict(winSize=(15,15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    
    # Filter valid points
    valid_prev = prev_pts[status.flatten() == 1]
    valid_curr = curr_pts[status.flatten() == 1]
    if len(valid_prev) == 0:
        return curr_img.copy(), (0, 0)
    
    # Compute average displacement
    displacements = valid_curr - valid_prev
    avg_disp = np.mean(displacements, axis=0)
    
    # Draw arrows on the current frame for each flow vector
    flow_vis = curr_img.copy()
    for p, q in zip(valid_prev, valid_curr):
        p = tuple(p.ravel().astype(int))
        q = tuple(q.ravel().astype(int))
        cv2.arrowedLine(flow_vis, p, q, (0, 0, 255), 2, tipLength=0.4)
    
    # Optionally, draw the head ROI for reference
    cv2.rectangle(flow_vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    return flow_vis, avg_disp

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
# Process and Visualize Keyframes Using Pose, YOLOv8, and Optical Flow
# ====================================================
def process_keyframes_visualization(folder_path, output_folder, yolo_model):
    """
    For each keyframe in the folder, this function:
      - Runs the pose-based head estimation and saves the overlay.
      - Runs YOLOv8 head detection and saves the overlay.
      - Computes optical flow for the head region between Address->Top and Top->Impact.
    Each visualization is saved separately in the output folder.
    """
    keyframes = load_keyframes(folder_path)
    if keyframes is None:
        print(f"[{folder_path}] Could not load keyframes.")
        return None
    os.makedirs(output_folder, exist_ok=True)
    results = {}
    
    # Compute pose overlays for individual keyframes
    for key, img in keyframes.items():
        pose_info = compute_pose_head_info(img)
        if pose_info is not None:
            vis_pose = visualize_pose_overlay(img, pose_info)
        else:
            print(f"[{folder_path}] Pose detection failed for {key}")
            vis_pose = img.copy()
        cv2.imwrite(os.path.join(output_folder, f"{key}_pose.jpg"), vis_pose)
        
        yolo_out = detect_head_yolov8(img, yolo_model)
        if yolo_out is not None:
            bbox, conf = yolo_out
            vis_yolo = overlay_yolo(img, bbox, conf)
        else:
            print(f"[{folder_path}] YOLO detection failed for {key}")
            vis_yolo = img.copy()
        cv2.imwrite(os.path.join(output_folder, f"{key}_yolo.jpg"), vis_yolo)
        
        results[key] = {"pose": vis_pose, "yolo": vis_yolo}
    
    # Compute optical flow between keyframe pairs (if pose info available)
    # For Address -> Top:
    pose_addr = compute_pose_head_info(keyframes["Address"])
    if pose_addr is not None:
        flow_AT, disp_AT = compute_optical_flow_head(keyframes["Address"], keyframes["Top"], pose_addr)
        cv2.imwrite(os.path.join(output_folder, "Address_Top_flow.jpg"), flow_AT)
        results["Address_Top_flow"] = {"flow": flow_AT, "avg_disp": disp_AT}
    else:
        print(f"[{folder_path}] Cannot compute optical flow for Address->Top (pose detection failed).")
    
    # For Top -> Impact:
    pose_top = compute_pose_head_info(keyframes["Top"])
    if pose_top is not None:
        flow_TI, disp_TI = compute_optical_flow_head(keyframes["Top"], keyframes["Impact"], pose_top)
        cv2.imwrite(os.path.join(output_folder, "Top_Impact_flow.jpg"), flow_TI)
        results["Top_Impact_flow"] = {"flow": flow_TI, "avg_disp": disp_TI}
    else:
        print(f"[{folder_path}] Cannot compute optical flow for Top->Impact (pose detection failed).")
    
    return results

# ====================================================
# Main Driver: Iterate Over Folders and Save Visualizations
# ====================================================
def main():
    top_level_dir = "event"  # Top-level directory with keyframe subfolders
    log_base = "logs"        # Base folder for visualizations
    os.makedirs(log_base, exist_ok=True)
    if not os.path.isdir(top_level_dir):
        print(f"Top-level directory '{top_level_dir}' does not exist!")
        return
    subfolders = [f for f in os.listdir(top_level_dir) if os.path.isdir(os.path.join(top_level_dir, f))]
    subfolders.sort()
    
    # Load YOLOv8 head detection model (ensure "yolov8n-head.pt" is available)
    yolo_model = YOLO("yolov8n-head.pt")
    
    for sub in subfolders:
        folder_path = os.path.join(top_level_dir, sub)
        log_folder = os.path.join(log_base, sub)
        os.makedirs(log_folder, exist_ok=True)
        print(f"\nProcessing folder: {folder_path}")
        res = process_keyframes_visualization(folder_path, log_folder, yolo_model)
        if res is not None:
            print(f"Visualizations for {sub} saved in {log_folder}")
        else:
            print(f"Processing failed for {folder_path}")

if __name__ == "__main__":
    main()
