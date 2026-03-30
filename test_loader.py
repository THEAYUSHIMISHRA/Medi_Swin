from torch.utils.data import DataLoader
from dataset.xray_dataset import XRayDataset
import config

dataset = XRayDataset(config.DATASET_PATH)

loader = DataLoader(
    dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)

batch = next(iter(loader))

print("Clean shape:", batch["clean"].shape)
print("Degraded shape:", batch["degraded"].shape)
print("Total images:", len(dataset))