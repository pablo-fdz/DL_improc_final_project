from .layers import DoubleConv, Down, Up, OutConv  # Importing custom layers for U-Net architecture
from torch import nn  # Importing PyTorch's neural network module

class UNet(nn.Module):

    """
    A U-Net architecture for image segmentation. It consists of an encoder path (downsampling) and a decoder path (upsampling),
    with skip connections between corresponding layers in the encoder and decoder paths.
    """

    def __init__(self, n_channels=3, n_classes=1, base_filters=8, dropout_p=0.1, bilinear=False):
        super().__init__()
        # define the encoder path
        self.inc   = DoubleConv(n_channels, base_filters, dropout_p)      # 3→8
        self.down1 = Down(base_filters,      base_filters*2, dropout_p)  # 8→16
        self.down2 = Down(base_filters*2,    base_filters*4, dropout_p)  # 16→32
        self.down3 = Down(base_filters*4,    base_filters*8, dropout_p)  # 32→64
        self.down4 = Down(base_filters*8,    base_filters*16, dropout_p) # 64→128 (bottleneck)

        # define the decoder path (mirrors encoder but in reverse)
        self.up1   = Up(base_filters*16, base_filters*8, dropout_p, bilinear) # 128→64
        self.up2   = Up(base_filters*8,  base_filters*4, dropout_p, bilinear) # 64→32
        self.up3   = Up(base_filters*4,  base_filters*2, dropout_p, bilinear) # 32→16
        self.up4   = Up(base_filters*2,  base_filters,   dropout_p, bilinear) # 16→8

        self.outc  = OutConv(base_filters, n_classes)  # 8→n_classes

        # Activation function for the output layer (only if loss compatible with probabilities and not only logits)
        # self.activation = nn.Sigmoid()  # final non-linearity for binary masks (convert logits to probabilities)
        # self.activation = nn.Softmax(dim=1)  # final non-linearity for multi-class masks (convert logits to probabilities)

    def forward(self, x):
        # ---- Encoder ----
        x1 = self.inc(x)      # 3→8 channels, full resolution
        x2 = self.down1(x1)   # 8→16 channels, 1/2ⁿ resolution
        x3 = self.down2(x2)   # 16→32
        x4 = self.down3(x3)   # 32→64
        x5 = self.down4(x4)   # 64→128 (bottleneck)

        # ---- Decoder with skip connections ----
        x  = self.up1(x5, x4) # decode 128→64, concat skip from x4
        x  = self.up2(x,  x3) # decode 64→32, concat skip from x3
        x  = self.up3(x,  x2) # decode 32→16, concat skip from x2
        x  = self.up4(x,  x1) # decode 16→8,  concat skip from x1

        logits = self.outc(x)        # 8→n_classes
        return logits  # Return logits (raw output of the network for compatibility with loss functions)
        # return self.activation(logits)  # apply final non-linearity (Sigmoid for binary or Softmax for multi-class)