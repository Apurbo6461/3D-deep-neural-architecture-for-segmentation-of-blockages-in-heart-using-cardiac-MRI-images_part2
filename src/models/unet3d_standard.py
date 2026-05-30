"""
Standard 4-level 3D U-Net (matches pretrained best_3d_u-net.pth).
Architecture: enc1(1->32) -> enc2(32->64) -> enc3(64->128) -> enc4(128->256)
-> bottleneck(256->512) -> dec4 -> dec3 -> dec2 -> dec1 -> final_conv.
Block structure: Conv3d(0), BN(1), ReLU(2), Conv3d(3), BN(4) so state_dict keys are enc1.0, enc1.1, enc1.3, enc1.4.
"""
import torch
import torch.nn as nn


def _double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_ch),
    )


class UNet3D(nn.Module):
    """4-level 3D U-Net compatible with best_3d_u-net.pth (base_filters=32)."""

    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()
        # Encoder
        self.enc1 = _double_conv(in_channels, base_filters)           # 1 -> 32
        self.pool1 = nn.MaxPool3d(2, 2)
        self.enc2 = _double_conv(base_filters, base_filters * 2)      # 32 -> 64
        self.pool2 = nn.MaxPool3d(2, 2)
        self.enc3 = _double_conv(base_filters * 2, base_filters * 4)  # 64 -> 128
        self.pool3 = nn.MaxPool3d(2, 2)
        self.enc4 = _double_conv(base_filters * 4, base_filters * 8)  # 128 -> 256
        self.pool4 = nn.MaxPool3d(2, 2)
        # Bottleneck
        self.bottleneck = _double_conv(base_filters * 8, base_filters * 16)  # 256 -> 512
        # Decoder
        self.up4 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = _double_conv(base_filters * 16, base_filters * 8)   # 512 -> 256
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = _double_conv(base_filters * 8, base_filters * 4)    # 256 -> 128
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = _double_conv(base_filters * 4, base_filters * 2)   # 128 -> 64
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = _double_conv(base_filters * 2, base_filters)       # 64 -> 32
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

    def _block_forward(self, block, x):
        out = block(x)
        return nn.functional.relu(out, inplace=True)

    def forward(self, x):
        e1 = self._block_forward(self.enc1, x)
        e2 = self._block_forward(self.enc2, self.pool1(e1))
        e3 = self._block_forward(self.enc3, self.pool2(e2))
        e4 = self._block_forward(self.enc4, self.pool3(e3))
        b = self._block_forward(self.bottleneck, self.pool4(e4))
        d4 = self._block_forward(self.dec4, torch.cat((self.up4(b), e4), dim=1))
        d3 = self._block_forward(self.dec3, torch.cat((self.up3(d4), e3), dim=1))
        d2 = self._block_forward(self.dec2, torch.cat((self.up2(d3), e2), dim=1))
        d1 = self._block_forward(self.dec1, torch.cat((self.up1(d2), e1), dim=1))
        return self.final_conv(d1)
