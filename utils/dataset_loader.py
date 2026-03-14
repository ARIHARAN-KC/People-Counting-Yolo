import os
import cv2
import torch
import numpy as np
import scipy.io
from torch.utils.data import Dataset
from .density_map import generate_density_map


class CrowdDataset(Dataset):

    def __init__(self, image_dir, gt_dir):

        self.image_dir = image_dir
        self.gt_dir = gt_dir

        self.images = []

        for img_name in os.listdir(image_dir):
            if img_name.endswith(".jpg"):
                self.images.append(img_name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)

        gt_name = img_name.replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
        gt_path = os.path.join(self.gt_dir, gt_name)

        img = cv2.imread(img_path)

        mat = scipy.io.loadmat(gt_path)

        points = mat["image_info"][0][0][0][0][0]

        density = generate_density_map(img, points)

        img = img / 255.0
        img = np.transpose(img, (2,0,1))

        img = torch.tensor(img).float()
        density = torch.tensor(density).unsqueeze(0).float()

        return img, density