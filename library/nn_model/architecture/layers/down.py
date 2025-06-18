from torch import nn
from .doubleconv import DoubleConv

class Down(nn.Module):

    """
    A module that performs a downsampling operation (by a factor of 2) using Max Pooling followed by a DoubleConv layer.
    """

    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),             # Halves height (H) and width (W) of the input
            DoubleConv(in_ch, out_ch, dropout_p)
        )
    def forward(self, x):
        return self.pool_conv(x)