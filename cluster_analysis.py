import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the metrics from the JSON file
with open("metrics.json", "r") as f:
    data = json.load(f)

# Build feature vectors with None for missing values
features = []
labels = []
for item in data:
    pose_disp = item.get("pose_max_disp_norm", None)
    yolo_disp = item.get("yolo_max_disp_norm", None)
    yolo_iou = item.get("yolo_avg_iou") if item.get("yolo_avg_iou") is not None else None
    flow_disp = item.get("optical_flow_max_disp_norm", None)
    features.append([pose_disp, yolo_disp, yolo_iou, flow_disp])
    labels.append(item["folder"])

features = np.array(features, dtype=np.float64)

# Step 1: Handle missing values (imputation)
imputer = SimpleImputer(strategy="mean")  # Replace missing values with feature-wise mean
features_imputed = imputer.fit_transform(features)

# Step 2: Standardize (normalize) the data
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features_imputed)

# Step 3: Run t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
features_2d = tsne.fit_transform(features_normalized)

# Step 4: Run K-Means Clustering with Multi-Start on t-SNE Features
kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
clusters_kmeans = kmeans.fit_predict(features_2d)

# Step 5: Determine which cluster is "good" (lower mean value in t-SNE space)
def assign_good_cluster(cluster_centers):
    return np.argmin(cluster_centers.mean(axis=1))  # Lower t-SNE cluster center means "good"

good_cluster_kmeans = assign_good_cluster(kmeans.cluster_centers_)
classification_kmeans = ["good" if c == good_cluster_kmeans else "bad" for c in clusters_kmeans]

# Step 6: Save classified results (handle NaN values in JSON)
classified_results = []
for i, label in enumerate(labels):
    classified_results.append({
        "folder": label,
        "pose_max_disp_norm": None if np.isnan(features[i, 0]) else features[i, 0],
        "yolo_max_disp_norm": None if np.isnan(features[i, 1]) else features[i, 1],
        "yolo_avg_iou": None if np.isnan(features[i, 2]) else features[i, 2],
        "optical_flow_max_disp_norm": None if np.isnan(features[i, 3]) else features[i, 3],
        "cluster_kmeans": int(clusters_kmeans[i]),
        "classification_kmeans": classification_kmeans[i]
    })

with open("classified_results.json", "w") as f:
    json.dump(classified_results, f, indent=2)

# Step 7: Plot results for K-Means with categorical colors
plt.figure(figsize=(8, 6))
colors = {"good": "blue", "bad": "red"}  # Define categorical colors
cluster_colors = [colors[classification] for classification in classification_kmeans]

scatter_kmeans = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_colors, s=100, alpha=0.75, edgecolors='k')

# Add enclosed labels below the points
for i, txt in enumerate(labels):
    plt.text(features_2d[i, 0], features_2d[i, 1] - 3, txt, 
             fontsize=8, ha='center', va='top', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

plt.title("t-SNE Visualization of Head Movement Metrics (K-Means)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.ylim(-45, None)
plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', label='Good', markerfacecolor='blue', markersize=10),
                    plt.Line2D([0], [0], marker='o', color='w', label='Bad', markerfacecolor='red', markersize=10)])
plt.tight_layout()
plt.savefig("clustering_results_kmeans.png")
plt.show()

print("Clustering results saved to 'classified_results.json' and visualization to 'clustering_results_kmeans.png'.")
