LightWeight Attention UNet for 3D volumetric medical image segmentation

Overview

This is a lightweight 3D Attention UNet variant.

Encoder optional: Depthwise Separable Convolution (DSConv)

Decoder optional: standard 3D convolution / DSConv

Attention Gate on skip connections

Flexible channel scaling via `feature\_scale`

Support 3D volumetric input (e.g., BraTS, multi-modal MRI)



