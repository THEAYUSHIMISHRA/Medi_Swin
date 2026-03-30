import torch
import os
from arch.generator import MediSwinGenerator
import config
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


def test_on_unseen(folder_type="Normal"):
    device = config.DEVICE
    model = MediSwinGenerator().to(device)

    # Load your "Gold" Checkpoint
    checkpoint = torch.load("checkpoints/ckpt_epoch_24.pth", map_location=device)
    state_dict = checkpoint['G_state_dict'] if 'G_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Match the training transforms
    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    # Pick an image from the "Normal" or "COVID" folder that WAS NOT in the training list
    test_path = os.path.join(config.DATASET_PATH, folder_type)
    test_files = os.listdir(test_path)
    # We pick the last few files (assuming training used the first 80%)
    sample_img_path = os.path.join(test_path, test_files[-1])

    img = Image.open(sample_img_path).convert("L")
    input_tensor = transform(img).unsqueeze(0).to(device).repeat(1, 3, 1, 1)

    with torch.no_grad():
        output = model(input_tensor)

    # Denormalize for viewing
    restored = torch.clamp((output + 1.0) / 2.0, 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()
    original = np.array(img.resize((224, 224)))

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1);
    plt.imshow(original, cmap='gray');
    plt.title(f"Unseen {folder_type} Input")
    plt.subplot(1, 2, 2);
    plt.imshow(restored);
    plt.title("Swin-GAN Output")
    plt.show()


if __name__ == "__main__":
    import numpy as np

    test_on_unseen(folder_type="Normal")  # Testing on healthy lungs is the ultimate validation