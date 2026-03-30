import torch
from dataset.xray_dataset import XRayDataset
from models.generator import MediSwinGenerator
from torch.utils.data import DataLoader
from utils.metrics import psnr, ssim_metric
from visualize import show_images
import config

device = config.DEVICE

dataset = XRayDataset(config.DATASET_PATH)

loader = DataLoader(dataset, batch_size=1)

model = MediSwinGenerator().to(device)
model.load_state_dict(torch.load("checkpoints/generator_epoch_19.pth"))
model.eval()

for batch in loader:

    clean = batch["clean"].to(device)
    degraded = batch["degraded"].to(device)

    with torch.no_grad():

        restored = model(degraded)

    print("PSNR:", psnr(restored, clean))
    print("SSIM:", ssim_metric(restored, clean))

    show_images(
        degraded[0],
        restored[0],
        clean[0]
    )

    break