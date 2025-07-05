import torch
import os
from torchvision import transforms
from PIL import Image

class LowLightDataset(torch.utils.data.Dataset):
    def __init__(self, low_path, high_path):
# Filter only image files (JPG, PNG, etc.)
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        self.low_images = sorted([f for f in os.listdir(low_path) if f.lower().endswith(valid_extensions)])
        self.high_images = sorted([f for f in os.listdir(high_path) if f.lower().endswith(valid_extensions)])

        self.low_path = low_path
        self.high_path = high_path

        # Transformation: Convert to tensor and normalize
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Converts [0,255] → [0,1] and uint8 → float32
        ])

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img_path = os.path.join(self.low_path, self.low_images[idx])
        high_img_path = os.path.join(self.high_path, self.high_images[idx])

        low_img = Image.open(low_img_path).convert("RGB")
        high_img = Image.open(high_img_path).convert("RGB")

        low_img = self.transform(low_img)  # Convert to float32 tensor
        high_img = self.transform(high_img)

        return low_img, high_img
