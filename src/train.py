import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LowLightDataset
from model import UNetEnhancer
import os

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths
DATASET_PATH = r"D:\LowLight-Enhancement\dataset\our485"
LOW_IMG_PATH = os.path.join(DATASET_PATH, "low")
HIGH_IMG_PATH = os.path.join(DATASET_PATH, "high")

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4

# Load dataset
dataset = LowLightDataset(LOW_IMG_PATH, HIGH_IMG_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, loss, and optimizer
model = UNetEnhancer().to(device)
criterion = nn.L1Loss()  # L1 Loss for image restoration
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for low_img, high_img in dataloader:
        low_img, high_img = low_img.to(device), high_img.to(device)

        optimizer.zero_grad()
        enhanced_img = model(low_img)
        loss = criterion(enhanced_img, high_img)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}")

# Save final model
torch.save(model.state_dict(), "lowlight_enhancer.pth")
print("✅ Model saved as 'lowlight_enhancer.pth'!")

