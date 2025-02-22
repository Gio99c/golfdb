import os
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

def detect_head_center_pose(image_bgr, landmarks_to_use=[0, 2, 5, 7, 8]):
    """
    Detects the head center using MediaPipe Pose.
    
    It uses the following landmarks:
      - 0: Nose
      - 2: Left eye (outer)
      - 5: Right eye (outer)
      - 7: Left ear
      - 8: Right ear

    Returns:
      (cx, cy) in pixel coordinates of the estimated head center,
      or None if pose landmarks cannot be detected.
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
    
    # Average the selected landmark coordinates (they are normalized)
    avg_x = sum(x for x, y in pts) / len(pts)
    avg_y = sum(y for x, y in pts) / len(pts)
    h, w, _ = image_bgr.shape
    cx = avg_x * w
    cy = avg_y * h
    return (cx, cy)

def compute_head_displacement(addr_img, imp_img):
    """
    Computes the displacement (in pixels) between the head center
    detected in the Address and Impact images using pose estimation.
    Returns:
      - displacement in pixels,
      - image width (to allow for normalization),
      or (None, width) if the head was not detected.
    """
    center_addr = detect_head_center_pose(addr_img)
    center_imp = detect_head_center_pose(imp_img)
    
    if center_addr is None or center_imp is None:
        return None, addr_img.shape[1]
    
    (cxA, cyA) = center_addr
    (cxI, cyI) = center_imp
    dist = np.sqrt((cxI - cxA)**2 + (cyI - cyA)**2)
    width = addr_img.shape[1]
    return dist, width

def analyze_swing_head(folder_path, displacement_threshold=0.1):
    """
    Given a folder containing key frames (e.g., Address.jpg and Impact.jpg),
    detects the head centers using MediaPipe Pose, computes the displacement,
    and classifies the swing as "good" if the normalized displacement is below
    the threshold (default 10% of image width), otherwise "bad".
    
    Returns a dictionary with the computed values and classification,
    or None if key frames are missing or head detection fails.
    """
    address_path = os.path.join(folder_path, "Address.jpg")
    impact_path  = os.path.join(folder_path, "Impact.jpg")
    
    if not os.path.exists(address_path) or not os.path.exists(impact_path):
        print(f"[{folder_path}] Missing Address or Impact frame. Skipping.")
        return None

    addr_img = cv2.imread(address_path)
    imp_img  = cv2.imread(impact_path)
    
    if addr_img is None or imp_img is None:
        print(f"[{folder_path}] Could not load Address or Impact image. Skipping.")
        return None

    displacement, width = compute_head_displacement(addr_img, imp_img)
    if displacement is None:
        print(f"[{folder_path}] Head not detected in one or both frames. Skipping.")
        return None

    disp_norm = displacement / width
    classification = "bad" if disp_norm > displacement_threshold else "good"

    return {
        "folder": folder_path,
        "displacement_px": displacement,
        "displacement_norm": disp_norm,
        "threshold": displacement_threshold,
        "classification": classification
    }

def main():
    """
    Iterates through subfolders of a top-level directory (each subfolder
    contains key frames from a video), computes head movement between
    Address and Impact using pose estimation, and prints the classification.
    """
    top_level_dir = "event"  # change this to the directory containing your key-frame folders
    if not os.path.isdir(top_level_dir):
        print(f"Top-level directory '{top_level_dir}' doesn't exist!")
        return

    subfolders = [f for f in os.listdir(top_level_dir)
                  if os.path.isdir(os.path.join(top_level_dir, f))]
    subfolders.sort()

    results = []
    for subf in subfolders:
        folder_path = os.path.join(top_level_dir, subf)
        print(f"\nAnalyzing folder: {folder_path}")
        result = analyze_swing_head(folder_path, displacement_threshold=0.1)
        if result:
            results.append(result)
            print(f"  => Swing classification: {result['classification']}, displacement_norm: {result['displacement_norm']:.3f}")
        else:
            print(f"  => Could not classify (missing frames or no head detected).")

    print("\n=== Final Results ===")
    for r in results:
        print(f"[{r['folder']}] => {r['classification']} | disp_norm={r['displacement_norm']:.3f} (threshold={r['threshold']})")

if __name__ == "__main__":
    main()