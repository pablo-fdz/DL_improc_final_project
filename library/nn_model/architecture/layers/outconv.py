from torch import nn

class OutConv(nn.Module):

    """
    A module that performs a 1x1 convolution to reduce the number of output channels to the number of classes.
    """

    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, n_classes, kernel_size=1)
    def forward(self, x):
        return self.conv(x)