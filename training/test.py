import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import scipy.io
from models.csrnet import CSRNet

# ----------------------------
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Load your trained model
checkpoint_path = "weights/csrnet_model_best.pth"
model = CSRNet()
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.")

# ----------------------------
# Paths
test_image_dir = "CrowdDataset/part_A_final/test_data/images"
test_gt_dir = "CrowdDataset/part_A_final/test_data/ground_truth"

# Get all image filenames
test_images = [f for f in os.listdir(test_image_dir) if f.endswith(".jpg")]
test_images.sort()

pred_counts = []
gt_counts = []

# ----------------------------
# Run inference
for img_name in test_images:
    img_path = os.path.join(test_image_dir, img_name)
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (512, 384))
    img_resized = img_resized / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))
    img_tensor = torch.tensor(img_resized).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_density = model(img_tensor)
        pred_density = F.interpolate(pred_density, size=(384, 512), mode='bilinear', align_corners=False)
        pred_count = pred_density.sum().item()
        pred_counts.append(pred_count)

    # Load ground truth
    gt_name = img_name.replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
    gt_path = os.path.join(test_gt_dir, gt_name)
    if os.path.exists(gt_path):
        mat = scipy.io.loadmat(gt_path)
        points = mat["image_info"][0][0][0][0][0]
        gt_counts.append(len(points))
    else:
        gt_counts.append(None)

    print(f"{img_name} - Predicted: {pred_count:.2f}, GT: {gt_counts[-1]}")

# ----------------------------
# Calculate metrics
valid_indices = [i for i, g in enumerate(gt_counts) if g is not None]
valid_pred = [pred_counts[i] for i in valid_indices]
valid_gt = [gt_counts[i] for i in valid_indices]

errors = [abs(p - g) for p, g in zip(valid_pred, valid_gt)]
squared_errors = [(p - g)**2 for p, g in zip(valid_pred, valid_gt)]

if errors:
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(squared_errors))
    total_gt = np.sum(valid_gt)
    accuracy = max(0, 100 * (1 - np.sum(errors) / total_gt))

    print("\n--- Final Metrics ---")
    print(f"Total Images Evaluated: {len(valid_indices)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Approximate Accuracy: {accuracy:.2f}%")