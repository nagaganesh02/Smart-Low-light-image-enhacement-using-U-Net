import torch
from model import UNetEnhancer  # Import model

# Initialize model
model = UNetEnhancer()

# Create a sample low-light image (Batch Size=1, 3 Channels, 256x256)
x = torch.randn(1, 3, 256, 256)

# Run model
y = model(x)

# Check output shape
print(f"Input Shape: {x.shape}, Output Shape: {y.shape}")
