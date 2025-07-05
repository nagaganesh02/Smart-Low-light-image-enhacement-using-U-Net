import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block for feature refinement
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual  # Skip connection

# Attention Mechanism for focus
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention  # Element-wise multiplication

# UNet Encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        pooled = self.pool(x)
        return pooled, x  # Return both pooled output and skip connection

# UNet Decoder
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)  # Upsampling
        x = torch.cat([x, skip], dim=1)  # Skip connection
        x = self.conv(x)
        return self.relu(x)

# Full UNet Model with Residual Blocks & Attention
class UNetEnhancer(nn.Module):
    def __init__(self):
        super(UNetEnhancer, self).__init__()
        self.encoder1 = Encoder(3, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)

        self.bottleneck = ResidualBlock(256)  # Middle block

        self.decoder3 = Decoder(256 + 256, 128)
        self.decoder2 = Decoder(128 + 128, 64)
        self.decoder1 = Decoder(64 + 64, 3)

        self.attention1 = AttentionBlock(64)
        self.attention2 = AttentionBlock(128)
        self.attention3 = AttentionBlock(256)

    def forward(self, x):
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)

        x_b = self.bottleneck(x3)

        x = self.decoder3(x_b, self.attention3(skip3))
        x = self.decoder2(x, self.attention2(skip2))
        x = self.decoder1(x, self.attention1(skip1))

        return torch.sigmoid(x)  # Output image in [0,1] range
