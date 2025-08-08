import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import collections

# -------------------- CONFIG --------------------
input_folder = "/home/vip-lab/mrudul/SAMKD/Dataset_drone_agri_512x512_4bands"
output_folder = "sam_dbscan_results"
sam_checkpoint = "/home/vip-lab/mrudul/SAMKD/checkpoints/sam_vit_b_01ec64.pth"  # Use vit_b for lower memory
model_type = "vit_b"  # Smaller model to reduce memory usage
device = "cuda" if torch.cuda.is_available() else "cpu"
dbscan_eps = 0.5  # DBSCAN: maximum distance for NDVI clustering
dbscan_min_samples = 3  # DBSCAN: minimum points to form a cluster
os.makedirs(output_folder, exist_ok=True)
print(f"Using device: {device}")

# -------------------- LOAD SAM --------------------
try:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
except RuntimeError as e:
    print(f"GPU memory error: {e}. Falling back to CPU.")
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,  # Reduced for memory efficiency
    pred_iou_thresh=0.9,  # Stricter to filter low-confidence masks
    stability_score_thresh=0.96,  # Stricter to filter unstable masks
    min_mask_region_area=200  # Larger to exclude small masks
)

# -------------------- PROCESS IMAGES --------------------
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".tif")])
clusters_used_per_image = []

for fname in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_folder, fname)

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] != 4:
        print(f"Skipping {fname}: Image must have 4 bands (Red, Green, Blue, NIR)")
        continue
    vis_image = image[:, :, :3]  # RGB for visualization
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    # Compute NDVI
    red = image[:, :, 0].astype(np.float32)
    nir = image[:, :, 3].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)  # Avoid division by zero

    # Generate SAM masks
    with torch.no_grad():  # Disable gradient computation for memory efficiency
        masks = mask_generator.generate(vis_image)
    print(f"{fname}: Generated {len(masks)} masks")
    if not masks:
        print(f"Skipping {fname}: No masks generated")
        continue

    # Extract features (mean NDVI)
    features = []
    for mask in masks:
        mask_binary = mask['segmentation'].astype(np.uint8)
        masked_ndvi = ndvi[mask_binary > 0]
        if len(masked_ndvi) > 0:
            mean_ndvi = np.mean(masked_ndvi)
        else:
            mean_ndvi = 0.0  # Fallback for empty masks
        features.append([mean_ndvi])
    features = np.array(features)

    if len(features) == 0:
        print(f"Skipping {fname}: No valid features")
        continue

    # Standardize features for DBSCAN
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    cluster_labels = dbscan.fit_predict(features_scaled)

    # Count clusters (excluding noise)
    unique_labels = np.unique(cluster_labels)
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    clusters_used_per_image.append(num_clusters)
    print(f"{fname}: DBSCAN found {num_clusters} clusters (excluding noise)")

    # Create semantic map (0 = background/noise, 1..N = clusters)
    semantic_map = np.zeros(vis_image.shape[:2], dtype=np.uint8)
    for idx, (mask, label) in enumerate(zip(masks, cluster_labels)):
        if label != -1:  # Ignore noise points
            semantic_map[mask['segmentation']] = label + 1  # Shift labels to start at 1

    # Save semantic label map
    outname = os.path.splitext(fname)[0] + ".png"
    outpath = os.path.join(output_folder, outname)
    cv2.imwrite(outpath, semantic_map)

    # Visualization
    colors = plt.cm.get_cmap("tab10", max(num_clusters + 1, 10))  # Include background
    colored_mask = colors(semantic_map / max(num_clusters + 1, 10))[:, :, :3]  # Drop alpha
    overlay = (0.5 * vis_image/255.0 + 0.5 * colored_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(vis_image)
    axes[0].set_title("Original Image")
    axes[1].imshow(semantic_map, cmap="tab10", vmin=0, vmax=max(num_clusters + 1, 10))
    axes[1].set_title(f"Semantic Map ({num_clusters} clusters)")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    vis_outname = os.path.splitext(fname)[0] + "_vis.png"
    vis_outpath = os.path.join(output_folder, vis_outname)
    plt.savefig(vis_outpath, bbox_inches="tight")
    plt.close(fig)

    # Clear GPU memory
    if device == "cuda":
        torch.cuda.empty_cache()

# -------------------- SUMMARY --------------------
cluster_summary = collections.Counter(clusters_used_per_image)
print("\nCluster usage summary across all images:")
for n, count in sorted(cluster_summary.items()):
    print(f"Images with {n} clusters: {count}")