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
    def __init__(self, input_channels=5):
        super(ResNetSeg, self).__init__()

        conv_params = {"kernel_size": 3, "stride": 2, "padding": 1}
        upsample_params = {"kernel_size": 2, "stride": 2, "padding": 0}
        conv7x7_params = {"kernel_size": 7, "stride": 1, "padding": 3}

        self.downsample = nn.Sequential(*[
            nn.Conv2d(input_channels, 16, **conv7x7_params),
            nn.Conv2d(16, 32, **conv_params),
            nn.Conv2d(32, 64, **conv_params)
        ])

        self.center_resblocks = nn.Sequential(*[
            PreActResidualBlock(64, 64, downsampling=False, bottleneck=False,
                                dropout=True)
            for i in range(16)
        ])

        self.upsample = nn.Sequential(*[
            nn.ConvTranspose2d(64, 32, **upsample_params),
            nn.ConvTranspose2d(32, 3, **upsample_params),
        ])
        self.out_conv = nn.Conv2d(3, 3, **conv7x7_params)

    def forward(self, x):
        downsampled = self.downsample(x)
        center = self.center_resblocks(downsampled)
        upsampled = self.upsample(center)
        return self.out_conv(upsampled)

if __name__ == "__main__":
    model = ResNetSeg(input_channels=5)
    # calculating # of parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total # of Params: {total}\nTrainable params: {trainable}")
