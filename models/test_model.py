import torch
from csrnet import CSRNet

model = CSRNet()

dummy = torch.randn(1,3,768,1024)

out = model(dummy)

print("Output shape:", out.shape)