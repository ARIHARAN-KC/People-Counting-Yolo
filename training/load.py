import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.csrnet import CSRNet   # fixed import

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load weights
model = CSRNet()
model.load_state_dict(torch.load("weights/csrnet_model_best.pth", map_location=device))
model.to(device)
model.eval()

print("Model loaded and ready for testing!")