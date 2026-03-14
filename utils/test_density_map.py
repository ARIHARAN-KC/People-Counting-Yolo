import os
import cv2
import scipy.io
import numpy as np
from density_map import generate_density_map

image_dir = "CrowdDataset/part_A_final/train_data/images"
gt_dir = "CrowdDataset/part_A_final/train_data/ground_truth"

total_actual = 0
total_predicted = 0
errors = []

# Get list of images
img_list = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

print(f"Starting processing for {len(img_list)} images...")

for img_name in img_list:
    image_path = os.path.join(image_dir, img_name)
    gt_name = img_name.replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
    gt_path = os.path.join(gt_dir, gt_name)

    # Load image and ground truth
    img = cv2.imread(image_path)
    mat = scipy.io.loadmat(gt_path)
    
    # Extract point coordinates
    points = mat["image_info"][0][0][0][0][0]
    actual_count = len(points)

    # Generate density map and calculate predicted count
    density = generate_density_map(img, points)
    predicted_count = density.sum()

    # Track totals and error
    total_actual += actual_count
    total_predicted += predicted_count
    errors.append(abs(actual_count - predicted_count))

# Final Statistics
mae = np.mean(errors)
rmse = np.sqrt(np.mean(np.square(errors)))

print("--- Final Results ---")
print(f"Total Images Processed: {len(img_list)}")
print(f"Total Actual People:   {total_actual}")
print(f"Total Predicted:       {round(total_predicted, 2)}")
print(f"Mean Absolute Error (MAE): {round(mae, 2)}")
print(f"Root Mean Square Error (RMSE): {round(rmse, 2)}")