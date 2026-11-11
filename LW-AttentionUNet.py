import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConv3d(nn.Module):
    """3D Depthwise Separable Convolution, reduce parameters and computation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleDSConv3d(nn.Module):
    """Two stacked 3D Depthwise Separable Convolution blocks"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            DSConv3d(in_ch, out_ch),
            DSConv3d(out_ch, out_ch)
        )

    def forward(self, x):
        return self.conv(x)


class DoubleConv3d(nn.Module):
    """Standard two stacked 3D ordinary convolution blocks"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_dsconv=True):
        super().__init__()
        if use_dsconv:
            self.conv = DoubleDSConv3d(in_ch, out_ch)
        else:
            self.conv = DoubleConv3d(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_dsconv=True):
        super().__init__()
        if use_dsconv:
            conv_block = DoubleDSConv3d(in_ch, out_ch)
        else:
            conv_block = DoubleConv3d(in_ch, out_ch)
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2, 2),
            conv_block
        )

    def forward(self, x):
        return self.mpconv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention Gate (AG) for attention UNet skip connection"""
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super().__init__()
        self.Wx = nn.Sequential(
            nn.Conv3d(in_channels_x, int_channels, kernel_size=1),
            nn.BatchNorm3d(int_channels)
        )
        self.Wg = nn.Sequential(
            nn.Conv3d(in_channels_g, int_channels, kernel_size=1),
            nn.BatchNorm3d(int_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(int_channels, 1, kernel_size=1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        x1 = self.Wx(x)
        g1 = self.Wg(g)
        attn_map = self.psi(F.relu(x1 + g1, inplace=True))
        return attn_map * x


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, out_channels, use_dsconv=False):
        super().__init__()
        self.attn_gate = AttentionGate(in_channels_x, in_channels_g, in_channels_g)

        if use_dsconv:
            self.conv_bn1 = DoubleDSConv3d(in_channels_g * 2, out_channels)
            self.conv_bn2 = DoubleDSConv3d(out_channels, out_channels)
        else:
            self.conv_bn1 = DoubleConv3d(in_channels_g * 2, out_channels)
            self.conv_bn2 = DoubleConv3d(out_channels, out_channels)

    def forward(self, x, x_skip):
        x = F.interpolate(x, x_skip.shape[2:], mode='trilinear', align_corners=False)
        x_att = self.attn_gate(x_skip, x)
        x = torch.cat([x_att, x], dim=1)
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        return x


class LWAttentionUNet(nn.Module):
    """
    LightWeight Attention UNet (LW-AttentionUNet) for 3D volumetric segmentation
    Args:
        in_channels: input modality channels
        num_classes: segmentation output channels
        feature_scale: scale factor to reduce base feature channels
        encoder_dsconv: whether use depthwise separable conv in encoder
        decoder_dsconv: whether use depthwise separable conv in decoder
    """
    def __init__(self, in_channels, num_classes, feature_scale=4,
                 encoder_dsconv=True, decoder_dsconv=False):
        super().__init__()
        base_features = [96, 192, 384, 768, 1280]
        self.features = [int(f / feature_scale) for f in base_features]

        self.inc = InConv(in_channels, self.features[0], use_dsconv=encoder_dsconv)
        self.down1 = Down(self.features[0], self.features[1], use_dsconv=encoder_dsconv)
        self.down2 = Down(self.features[1], self.features[2], use_dsconv=encoder_dsconv)
        self.down3 = Down(self.features[2], self.features[3], use_dsconv=encoder_dsconv)
        self.down4 = Down(self.features[3], self.features[3], use_dsconv=encoder_dsconv)

        self.up1 = AttentionUpBlock(self.features[3], self.features[3], self.features[2], use_dsconv=decoder_dsconv)
        self.up2 = AttentionUpBlock(self.features[2], self.features[2], self.features[1], use_dsconv=decoder_dsconv)
        self.up3 = AttentionUpBlock(self.features[1], self.features[1], self.features[0], use_dsconv=decoder_dsconv)
        self.up4 = AttentionUpBlock(self.features[0], self.features[0], self.features[0], use_dsconv=decoder_dsconv)
        self.outc = OutConv(self.features[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    model = LWAttentionUNet(in_channels=4, num_classes=4)
    x = torch.rand(1, 4, 128, 128, 64).to(device)

    flops = FlopCountAnalysis(model, x)
    print("FLOPs (fvcore):", flops.total())
    print(parameter_count_table(model))
    print("Total params:", sum(p.numel() for p in model.parameters()))
