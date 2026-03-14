from dataset_loader import CrowdDataset

image_dir = "CrowdDataset/part_A_final/train_data/images"
gt_dir = "CrowdDataset/part_A_final/train_data/ground_truth"

dataset = CrowdDataset(image_dir, gt_dir)

img, density = dataset[0]

print("Image shape:", img.shape)
print("Density shape:", density.shape)
print("Count:", density.sum().item())