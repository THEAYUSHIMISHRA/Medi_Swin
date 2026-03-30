import torch
from models.discriminator import PatchGANDiscriminator

model = PatchGANDiscriminator()

x = torch.randn(1,3,224,224)

y = model(x)

print("Input:", x.shape)
print("Output:", y.shape)