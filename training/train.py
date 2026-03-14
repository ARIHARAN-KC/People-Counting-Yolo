import os
import sys
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import scipy.io

from models.csrnet import CSRNet
from utils.density_map import generate_density_map

# ----------------------------
# Dataset with augmentation
# ----------------------------
class CrowdDataset(Dataset):
    def __init__(self, image_dir, gt_dir, transform=True, resize=(512, 384)):
        self.image_dir = image_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.resize = resize
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        gt_name = img_name.replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
        gt_path = os.path.join(self.gt_dir, gt_name)

        img = cv2.imread(img_path)
        if self.resize:
            img = cv2.resize(img, self.resize)

        mat = scipy.io.loadmat(gt_path)
        points = mat["image_info"][0][0][0][0][0]

        density = generate_density_map(img, points)

        # Data augmentation: horizontal flip (with copy to fix negative stride)
        if self.transform and random.random() > 0.5:
            img = cv2.flip(img, 1).copy()
            density = np.flip(density, 1).copy()

        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        img = torch.tensor(img).float()
        density = torch.tensor(density).unsqueeze(0).float()

        return img, density

# ----------------------------
# Paths
# ----------------------------
image_dir = "CrowdDataset/part_A_final/train_data/images"
gt_dir = "CrowdDataset/part_A_final/train_data/ground_truth"
weights_dir = "weights"
os.makedirs(weights_dir, exist_ok=True)

# ----------------------------
# Dataset and DataLoader
# ----------------------------
dataset = CrowdDataset(image_dir, gt_dir, transform=True, resize=(512, 384))
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# Model, Loss, Optimizer, Scheduler
# ----------------------------
model = CSRNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ----------------------------
# Training with Early Stopping
# ----------------------------
epochs = 15        # max epochs
patience = 4       # early stopping patience
best_loss = float("inf")
counter = 0

for epoch in range(epochs):
    total_loss = 0.0
    model.train()

    for batch_idx, (img, density) in enumerate(loader):
        img = img.to(device)
        density = density.to(device)

        pred = model(img)
        density_resized = F.interpolate(
            density, size=pred.shape[2:], mode='bilinear', align_corners=False
        )

        loss = criterion(pred, density_resized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Print batch progress every 10 images
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.6f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.6f}")

    # Scheduler step
    scheduler.step(avg_loss)

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), os.path.join(weights_dir, "csrnet_model_best.pth"))
        print("Best model saved.")
    else:
        counter += 1
        print(f"No improvement for {counter} epoch(s).")

    if counter >= patience:
        print(f"Early stopping triggered after {patience} epochs with no improvement.")
        break

# Save final model
torch.save(model.state_dict(), os.path.join(weights_dir, "csrnet_model_final.pth"))
print("Final model saved.")