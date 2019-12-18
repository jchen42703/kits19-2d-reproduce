import torch.nn as nn

from .neural_network import SegmentationNetwork
from .utils import PreActResidualBlock

class ResNetSeg(SegmentationNetwork):
    """
    Used in Stage 2. Authors attribute it to this paper:
        van Harten, L., Noothout, J.M., Verhoeff, J., Wolterink, J.M.,
        Isgum, I.: Automatic segmentation of organs at risk in thoracic ct
        scans by combining 2d and 3d convolutional neural networks.
        In: SegTHOR@ ISBI. (2019)
    """
    def __init__(self, channels_in=5):
        conv_params = {"kernel_size": 3, "stride": 2, "padding": 1}
        upsample_params = {"kernel_size": 2, "stride": 2, "padding": 1}

        self.downsample = nn.Sequential(*[
            nn.Conv2d(channels_in, 16, kernel_size=7, stride=1, padding=1),
            nn.Conv2d(16, 32, **conv_params),
            nn.Conv2d(32, 64, **conv_params)
        ])

        self.center_resblocks = nn.Sequential(*[
            PreActResidualBlock(64, 64, downsampling=False, bottleneck=False)
            for i in range(16)
        ])

        self.upsample = nn.Sequential(*[
            nn.ConvTranspose2d(64, 32, **upsample_params),
            nn.ConvTranspose2d(32, 3, **upsample_params),
        ])
        self.out_conv = nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        downsampled = self.downsample(x)
        center = self.center_resblocks(downsampled)
        upsampled = self.upsample(center)
        return self.out_conv(upsampled)
