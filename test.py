import torch
import torchvision.transforms as T
from PIL import Image
from arch.generator import MediSwinGenerator
import config
import os


def test_single_image(image_path, checkpoint_path):
    device = config.DEVICE

    # 1. Load Model
    model = MediSwinGenerator().to(device)

    # Use weights_only=True for security and compatibility
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if the checkpoint is a full dictionary or just state_dict
    if 'G_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['G_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. Preprocess Image (Aligned with Training)
    transform = T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),  # Must be 224
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    input_img = Image.open(image_path).convert("L")
    input_tensor = transform(input_img).unsqueeze(0).to(device)

    # Crucial: Match the 3-channel repeat used in training
    input_tensor = input_tensor.repeat(1, 3, 1, 1)

    # 3. Inference
    with torch.no_grad():
        restored_tensor = model(input_tensor)

    # 4. Post-process and Save
    # Denormalize: [-1, 1] -> [0, 1]
    restored_tensor = (restored_tensor + 1) / 2.0
    restored_tensor = torch.clamp(restored_tensor, 0, 1)

    # Save as Grayscale if you prefer, or RGB to see the 3-channel output
    restored_img = T.ToPILImage()(restored_tensor.squeeze(0).cpu())

    save_path = f"results/restored_{os.path.basename(checkpoint_path)}_{os.path.basename(image_path)}"
    os.makedirs("results", exist_ok=True)
    restored_img.save(save_path)
    print(f"Saved restored image to: {save_path}")


if __name__ == "__main__":
    # Update these paths
    TEST_IMAGE = r"E:\Ayushi\Degradation_Aware_Media_Swin\data\raw\COVID\sample.png"
    # Testing your stable Epoch 18 vs the latest
    CHECKPOINT = "checkpoints/ckpt_epoch_18.pth"

    if os.path.exists(CHECKPOINT) and os.path.exists(TEST_IMAGE):
        test_single_image(TEST_IMAGE, CHECKPOINT)
    else:
        print("Check your paths! Either the image or the checkpoint is missing.")