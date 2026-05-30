import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard convolution block: Conv3D -> BN -> ReLU -> Conv3D -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    """
    Residual Block: Two 3D convolutions with a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    """
    Attention Gate to focus on relevant features in the skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet3D(nn.Module):
    """
    Unified 3D U-Net Model.
    Supports:
    - Standard 3D U-Net
    - Residual 3D U-Net (use_residual=True)
    - Attention 3D U-Net (use_attention=True)
    - Residual-Attention 3D U-Net (use_residual=True, use_attention=True)
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=16, use_residual=False, use_attention=False):
        super(UNet3D, self).__init__()
        
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Block type selection
        Block = ResBlock if use_residual else ConvBlock
        
        # Encoder
        self.enc1 = Block(in_channels, base_filters)
        self.pool1 = nn.MaxPool3d(2, 2)
        
        self.enc2 = Block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool3d(2, 2)
        
        self.enc3 = Block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool3d(2, 2)
        
        # Bottleneck (added extra depth to match ResAtt architecture more closely if needed, 
        # but sticking to 3 levels + bottleneck for consistency with previous ResAtt implementation)
        self.bottleneck = Block(base_filters * 4, base_filters * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        if self.use_attention:
            self.att3 = AttentionBlock(F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2)
        self.dec3 = Block(base_filters * 8, base_filters * 4) 
        
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        if self.use_attention:
            self.att2 = AttentionBlock(F_g=base_filters * 2, F_l=base_filters * 2, F_int=base_filters)
        self.dec2 = Block(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        if self.use_attention:
            self.att1 = AttentionBlock(F_g=base_filters, F_l=base_filters, F_int=base_filters // 2)
        self.dec1 = Block(base_filters * 2, base_filters)
        
        # Final Output
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        
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
        
        # Decoder
        d3 = self.up3(b)
        if self.use_attention:
            x3 = self.att3(g=d3, x=e3)
            d3 = torch.cat((x3, d3), dim=1)
        else:
            d3 = torch.cat((e3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if self.use_attention:
            x2 = self.att2(g=d2, x=e2)
            d2 = torch.cat((x2, d2), dim=1)
        else:
            d2 = torch.cat((e2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if self.use_attention:
            x1 = self.att1(g=d1, x=e1)
            d1 = torch.cat((x1, d1), dim=1)
        else:
            d1 = torch.cat((e1, d1), dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return out


class ResAttUNet3D(UNet3D):
    """Residual-Attention 3D U-Net (use_residual=True, use_attention=True)."""

    def __init__(self, in_channels=1, out_channels=1, base_filters=16):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=base_filters,
            use_residual=True,
            use_attention=True,
        )


if __name__ == "__main__":
    # Test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Unified UNet3D on {device}")
    
    # 1. Standard U-Net
    model_std = UNet3D(use_residual=False, use_attention=False).to(device)
    print("Standard U-Net initialized")
    
    # 2. ResAtt U-Net
    model_resatt = UNet3D(use_residual=True, use_attention=True).to(device)
    print("ResAtt U-Net initialized")
    
    x = torch.randn(1, 1, 64, 64, 64).to(device)
    with torch.no_grad():
        y_std = model_std(x)
        y_resatt = model_resatt(x)
    
    print(f"Standard Output: {y_std.shape}")
    print(f"ResAtt Output: {y_resatt.shape}")
    print("✓ Verification successful")
