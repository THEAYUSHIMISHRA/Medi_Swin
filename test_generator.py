import torch
from arch.generator import MediSwinGenerator

model = MediSwinGenerator()

x = torch.randn(1,3,224,224)

y = model(x)

print("Input:", x.shape)
print("Output:", y.shape)