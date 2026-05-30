import torch
import torch.nn as nn
import torch.nn.functional as F

class VNet3D(nn.Module):
    """
    3D V-Net architecture for medical image segmentation.
    V-Net uses residual connections and a different structure than U-Net.
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super(VNet3D, self).__init__()
        
        # Encoder
        self.enc1 = self._res_block(in_channels, base_filters)
        self.pool1 = nn.Conv3d(base_filters, base_filters * 2, kernel_size=2, stride=2)
        
        self.enc2 = self._res_block(base_filters * 2, base_filters * 2)
        self.pool2 = nn.Conv3d(base_filters * 2, base_filters * 4, kernel_size=2, stride=2)
        
        self.enc3 = self._res_block(base_filters * 4, base_filters * 4)
        self.pool3 = nn.Conv3d(base_filters * 4, base_filters * 8, kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._res_block(base_filters * 8, base_filters * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self._res_block(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self._res_block(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self._res_block(base_filters * 2, base_filters)
        
        # Final output
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        
    def _res_block(self, in_channels, out_channels):
        """Residual block with two convolutions."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.InstanceNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.InstanceNorm3d(out_channels),
            nn.PReLU()
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return out
