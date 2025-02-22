import os
import cv2
import numpy as np
import mediapipe as mp

# -----------------------------
# Method 1: Pose-based detection
# -----------------------------
mp_pose = mp.solutions.pose

def detect_head_center_pose(image_bgr, landmarks_to_use=[0, 2, 5, 7, 8]):
    """
    Uses MediaPipe Pose to estimate head center.
    Selected landmarks: 
      - 0: Nose, 2: Left eye outer, 5: Right eye outer,
      - 7: Left ear, 8: Right ear.
    Returns:
      (cx, cy) in pixel coordinates, or None if not detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None
    landmarks = results.pose_landmarks.landmark
    pts = []
    for i in landmarks_to_use:
        pt = landmarks[i]
        pts.append((pt.x, pt.y))
    if not pts:
        return None
    avg_x = sum(x for x, y in pts) / len(pts)
    avg_y = sum(y for x, y in pts) / len(pts)
    h, w, _ = image_bgr.shape
    return (avg_x * w, avg_y * h)

# -----------------------------
# Method 2: Face Mesh–based detection
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

def detect_head_center_facemesh(image_bgr, min_confidence=0.3):
    """
    Uses MediaPipe Face Mesh to detect dense facial landmarks.
    Returns the average (x,y) in pixels of the detected landmarks,
    or None if no face is detected.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                min_detection_confidence=min_confidence) as face_mesh:
        results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        return None
    face_landmarks = results.multi_face_landmarks[0].landmark
    pts = []
    h, w, _ = image_bgr.shape
    for lm in face_landmarks:
        pts.append((lm.x * w, lm.y * h))
    if not pts:
        return None
    pts = np.array(pts)
    cx, cy = np.mean(pts, axis=0)
    return (cx, cy)

# -----------------------------
# Helper: Compute distances between keypoints
# -----------------------------
def compute_displacement(p1, p2):
    """Compute Euclidean distance between two (x,y) points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_head_displacements(keyframes, detection_fn):
    """
    Given a dict of keyframes with keys: "Address", "Top", "Impact"
    and a detection function (e.g. detect_head_center_pose or detect_head_center_facemesh),
    compute the normalized distances between:
      - Address and Top (A-T)
      - Top and Impact (T-I)
      - Address and Impact (A-I)
    Normalization is done by dividing the pixel distance by the image width.
    Returns a dictionary with the displacements (both in pixels and normalized)
    or None if detection fails for any key frame.
    """
    required = ["Address", "Top", "Impact"]
    centers = {}
    for key in required:
        if key not in keyframes:
            print(f"Missing keyframe: {key}")
            return None
        img = keyframes[key]
        center = detection_fn(img)
        if center is None:
            print(f"Detection failed for {key}")
            return None
        centers[key] = center
    # Use the width of the Address image for normalization
    width = keyframes["Address"].shape[1]
    
    d_AT = compute_displacement(centers["Address"], centers["Top"])
    d_TI = compute_displacement(centers["Top"], centers["Impact"])
    d_AI = compute_displacement(centers["Address"], centers["Impact"])
    
    return {
        "A-T_px": d_AT, "A-T_norm": d_AT / width,
        "T-I_px": d_TI, "T-I_norm": d_TI / width,
        "A-I_px": d_AI, "A-I_norm": d_AI / width,
        "avg_norm": (d_AT + d_TI + d_AI) / (3 * width)
    }

# -----------------------------
# Wrapper to load key frames from a folder
# -----------------------------
def load_keyframes(folder_path):
    """
    Expects the folder to contain key frames with filenames
    like "Address.jpg", "Top.jpg", "Impact.jpg".
    Returns a dictionary mapping event name to image (BGR).
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

# -----------------------------
# Analysis function that computes metrics from both methods
# -----------------------------
def analyze_swing_head_advanced(folder_path, threshold=0.1):
    """
    Given a folder containing key frames, this function:
      - Loads the Address, Top, and Impact frames.
      - Computes head displacement metrics using two methods:
          * Pose-based
          * Face Mesh–based
      - For each method, computes distances for Address-Top, Top-Impact, and Address-Impact
      - Also computes an average normalized displacement.
      - Classifies the swing as "bad" if the average normalized displacement exceeds the threshold.
    Returns a dictionary with all the metrics.
    """
    keyframes = load_keyframes(folder_path)
    if keyframes is None:
        print(f"[{folder_path}] Could not load key frames.")
        return None

    # Get displacements from pose method
    pose_metrics = compute_head_displacements(keyframes, detect_head_center_pose)
    # Get displacements from face mesh method
    facemesh_metrics = compute_head_displacements(keyframes, lambda img: detect_head_center_facemesh(img, min_confidence=0.3))
    
    if pose_metrics is None and facemesh_metrics is None:
        print(f"[{folder_path}] Both methods failed.")
        return None

    # Define classification based on each method
    result = {"folder": folder_path}
    if pose_metrics:
        result["pose"] = pose_metrics
        result["pose_classification"] = "bad" if pose_metrics["avg_norm"] > threshold else "good"
    else:
        result["pose"] = None
        result["pose_classification"] = "unknown"
    
    if facemesh_metrics:
        result["facemesh"] = facemesh_metrics
        result["facemesh_classification"] = "bad" if facemesh_metrics["avg_norm"] > threshold else "good"
    else:
        result["facemesh"] = None
        result["facemesh_classification"] = "unknown"
    
    # Optionally, combine the two metrics (if both available, average the normalized displacements)
    if pose_metrics and facemesh_metrics:
        combined_avg = (pose_metrics["avg_norm"] + facemesh_metrics["avg_norm"]) / 2.0
        result["combined_avg_norm"] = combined_avg
        result["combined_classification"] = "bad" if combined_avg > threshold else "good"
    else:
        result["combined_avg_norm"] = None
        result["combined_classification"] = "unknown"
    
    return result

# -----------------------------
# Main driver: iterate over subfolders
# -----------------------------
def main():
    """
    Iterates through subfolders in a top-level directory (each subfolder contains key frames),
    computes head movement using both pose-based and face mesh–based methods (with three distance metrics),
    and prints the results.
    """
    top_level_dir = "event"  # adjust as needed
    if not os.path.isdir(top_level_dir):
        print(f"Top-level dir '{top_level_dir}' doesn't exist!")
        return

    subfolders = [f for f in os.listdir(top_level_dir) if os.path.isdir(os.path.join(top_level_dir, f))]
    subfolders.sort()

    results = []
    for sub in subfolders:
        folder_path = os.path.join(top_level_dir, sub)
        print(f"\nAnalyzing folder: {folder_path}")
        res = analyze_swing_head_advanced(folder_path, threshold=0.1)
        if res:
            results.append(res)
            print(f"  Pose method: {res['pose_classification']} (avg_norm={res['pose']['avg_norm']:.3f}" if res["pose"] else "N/A")
            print(f"  Face Mesh method: {res['facemesh_classification']} (avg_norm={res['facemesh']['avg_norm']:.3f}" if res["facemesh"] else "N/A")
            if res["combined_avg_norm"] is not None:
                print(f"  Combined: {res['combined_classification']} (avg_norm={res['combined_avg_norm']:.3f})")
        else:
            print("  Could not analyze this folder.")

    # Optionally, print a summary of results
    print("\n=== Final Summary ===")
    for r in results:
        print(f"[{r['folder']}] => Pose: {r['pose_classification']}, FaceMesh: {r['facemesh_classification']}, Combined: {r['combined_classification']}")
    
if __name__ == "__main__":
    main()
