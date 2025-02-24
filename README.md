# **Head Movement Analysis in Golf Swings - Custom Scripts**

Below is a brief overview of the **additional** scripts that expand on the original GolfDB repository. These scripts handle video preprocessing, head-movement analysis, and unsupervised clustering for swing classification. For core information on GolfDB and SwingNet usage (training, evaluating, etc.), please refer to the [original GolfDB repository](https://github.com/wmcnally/golfdb).

---

## **1. Preprocessing Videos: `preprocess_custom.py`**

### **Description**
- Resizes and center-crops raw videos to **160x160** resolution.
- Produces output videos ready for **SwingNet** inference or custom analysis.

### **Usage**
- Place raw videos in `data/kaggle_blurred/`.
- The script outputs processed videos in `data/videos_160/`.

### **Run the script**
```bash
python preprocess_custom.py
```

---

## **2. Extracting Head Movement Metrics: `head_analysis.py`**

### **Description**
- Analyzes **Address**, **Top**, and **Impact** frames to derive:
  1. **Pose-based** head displacement.
  2. **YOLO-based** head bounding box displacement + IoU.
  3. **Optical flow** displacement in the head region.
- Saves results as a JSON file (`metrics.json`).

### **Input Requirements**
- A folder structure under `event/` with subfolders for each swing, containing:
  - `Address.jpg`
  - `Top.jpg`
  - `Impact.jpg`
- A pretrained YOLO model (e.g., `yolov8n-head.pt`).

### **Run the script**
```bash
python head_analysis.py
```

**Output**:
- `metrics.json` containing per-swing metrics.

---

## **3. Visualizing Head Movement: `head_analysis_visualize.py`**

### **Description**
- Generates **visual overlays**:
  - Pose landmarks (shoulders, nose, refined head).
  - YOLO bounding box.
  - Optical flow arrows between frames.
- Saves annotated images under `logs/`.

### **Run the script**
```bash
python head_analysis_visualize.py
```

**Output**:
- In `logs/<subfolder>`, you get files like `Address_pose.jpg`, `Top_yolo.jpg`, `Address_Top_flow.jpg`, etc.

---

## **4. Clustering and Classification: `cluster_analysis.py`**

### **Description**
- Reads `metrics.json`.
- Imputes missing values (e.g., YOLO fails in some frames).
- Standardizes numeric features.
- Runs **t-SNE** for 2D visualization.
- Performs **k-means** clustering (`k=2`) to separate “good” vs. “bad” swings.
- Saves results to `classified_results.json` and a plot `clustering_results_kmeans.png`.

### **Run the script**
```bash
python cluster_analysis.py
```

**Output**:
- `classified_results.json`
- `clustering_results_kmeans.png` (t-SNE + cluster assignments)

---

## **Summary of Scripts**

| **Script**               | **Purpose**                                                         | **Output**                                |
|--------------------------|---------------------------------------------------------------------|-------------------------------------------|
| `preprocess_custom.py`   | Resize & center-crop videos to 160x160                              | `data/videos_160/` (processed videos)     |
| `head_analysis.py`       | Compute pose, YOLO, & optical flow metrics for Address–Top–Impact   | `metrics.json`                            |
| `head_analysis_visualize.py` | Generate overlay images for pose, detection, and flow                | Visual files under `logs/`                |
| `cluster_analysis.py`    | Cluster swings (via t-SNE + k-means) into good/bad categories       | `classified_results.json`, cluster plots  |

Use these scripts to **preprocess your data, track head movement, and unsupervisedly categorize** golf swings based on head stability.
