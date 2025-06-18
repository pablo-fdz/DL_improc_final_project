from torch import nn

class DoubleConv(nn.Module):

    """
    A module that performs two consecutive 3x3 convolutions, each followed by Batch Normalization and ReLU activation.
    It also includes a spatial dropout layer to randomly drop out whole feature maps, which helps regularize the representations.
    """

    def __init__(self, in_ch, out_ch, dropout_p=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 1st 3×3 conv → BatchNorm → ReLU
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            # 2nd 3×3 conv → BatchNorm → ReLU
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            # Spatial dropout to randomly drop out whole feature maps - regularizes the representations
            nn.Dropout2d(p=dropout_p)  # Spatial dropout with the proportion dropout_p
        )
    def forward(self, x):
        return self.double_conv(x)