import torch
from torch import nn
from .doubleconv import DoubleConv

class Up(nn.Module):

    """
    A module that performs an upsampling operation (by a factor of 2) using either bilinear interpolation or transposed convolution,
    followed by a DoubleConv layer. It also concatenates the upsampled features with the corresponding skip connection from the encoder.
    """

    def __init__(self, in_ch, out_ch, dropout_p=0.1, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, dropout_p)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            # after upconv, channels=out_ch; concatenated with skip (also out_ch) → 2*out_ch
            self.conv = DoubleConv(out_ch * 2, out_ch, dropout_p)
    def forward(self, x, skip):

        x = self.up(x)               # upsample the coarse features

        # Pad if needed to match the encoder feature size
        # Sometimes the upsampled feature map might have a different spatial size
        # compared to the corresponding skip connection
        if x.size() != skip.size():
            diff_h = skip.size()[2] - x.size()[2]  # height difference
            diff_w = skip.size()[3] - x.size()[3]  # width difference
            # Pad the upsampled feature map to match the skip connection size
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2, 
                                    diff_h // 2, diff_h - diff_h // 2])

        # Concatenate the upsampled features with the corresponding skip connection
        # along the channel dimension (dim=1)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)