import os
import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def detect_head_center_mediapipe(image_bgr):
    """
    Detects a face in the input BGR image using MediaPipe FaceDetection.
    Returns (cx, cy) for the face bounding box center in pixel coordinates,
    or None if no face detected.
    """
    # Convert BGR -> RGB for MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # By default, FaceDetection confidence threshold is 0.5
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_det:
        results = face_det.process(image_rgb)

    if not results.detections:
        return None

    # For simplicity, just take the first detected face
    detection = results.detections[0]
    # The detection.location_data.relative_bounding_box 
    # gives bounding box in normalized [0,1] coordinates
    bbox = detection.location_data.relative_bounding_box

    # Convert to pixel coords
    h, w, _ = image_bgr.shape
    bbox_cx = (bbox.xmin + bbox.width / 2) * w
    bbox_cy = (bbox.ymin + bbox.height / 2) * h

    return (bbox_cx, bbox_cy)

def compute_head_displacement(addr_img, imp_img):
    """
    Detect head center in Address image vs. Impact image.
    Returns a displacement in pixels and the frame width (for optional normalization).
    """
    center_addr = detect_head_center_mediapipe(addr_img)
    center_imp = detect_head_center_mediapipe(imp_img)

    if center_addr is None or center_imp is None:
        # If face not found in either frame, return None
        return None, addr_img.shape[1]

    (cxA, cyA) = center_addr
    (cxI, cyI) = center_imp

    dist = np.sqrt((cxI - cxA)**2 + (cyI - cyA)**2)
    width = addr_img.shape[1]
    return dist, width

def analyze_swing_head(folder_path, displacement_threshold=0.1):
    """
    Given a folder containing key frames:
      e.g. Address.jpg, Impact.jpg, ...
    Attempts to detect the face in Address and Impact,
    measures displacement, and classifies 'good' vs 'bad'.
    - displacement_threshold is fraction of image width.
      e.g. 0.1 => 10% of the frame width
    Returns a dictionary with info or None if frames missing.
    """

    address_path = os.path.join(folder_path, "Address.jpg")
    impact_path  = os.path.join(folder_path, "Impact.jpg")

    # If we don’t have both frames, skip or return None
    if not os.path.exists(address_path) or not os.path.exists(impact_path):
        print(f"[{folder_path}] Missing Address or Impact frame. Skipping.")
        return None

    addr_img = cv2.imread(address_path)
    imp_img  = cv2.imread(impact_path)

    # If images are not loaded
    if addr_img is None or imp_img is None:
        print(f"[{folder_path}] Could not load Address or Impact image. Skipping.")
        return None

    displacement, width = compute_head_displacement(addr_img, imp_img)
    if displacement is None:
        print(f"[{folder_path}] Face not found in one or both frames. Skipping.")
        return None

    # Normalized displacement (fraction of image width)
    disp_norm = displacement / width

    # Compare to threshold
    # e.g. if disp_norm > 0.1 => "bad" else "good"
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
    Example driver script that iterates through a top-level directory,
    finds subfolders (each presumably containing key frames),
    and runs head analysis on each.
    """

    top_level_dir = "event" 
    # Make sure this directory exists
    if not os.path.isdir(top_level_dir):
        print(f"Top-level dir {top_level_dir} doesn't exist!")
        return

    # List subfolders
    subfolders = [f for f in os.listdir(top_level_dir)
                  if os.path.isdir(os.path.join(top_level_dir, f))]
    subfolders.sort()

    # We'll store results in a list
    results = []

    for subf in subfolders:
        folder_path = os.path.join(top_level_dir, subf)
        print(f"\nAnalyzing folder: {folder_path}")
        result = analyze_swing_head(folder_path, displacement_threshold=0.1)
        if result:
            results.append(result)
            print(f"  => Swing classification: {result['classification']}, displacement= {result['displacement_norm']:.3f} (fraction of width)")
        else:
            print(f"  => Could not classify (missing frames or no face).")

    # Summarize
    print("\n=== Final Results ===")
    for r in results:
        print(f"[{r['folder']}] => {r['classification']} | disp_norm={r['displacement_norm']:.3f} threshold={r['threshold']}")


if __name__ == "__main__":
    main()
