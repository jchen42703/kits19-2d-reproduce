import torch.nn as nn
import torch

class BNReLUConv2d(nn.Module):
    """
    Batch Normalization -> ReLU -> Conv2d
    """
    def __init__(self, input_feature_channels, output_feature_channels,
                 conv_stride=1):

        super(BNReLUConv2d, self).__init__()
        norm_op_kwargs = {"eps": 1e-5, "affine": True, "momentum": 0.1,
                          "track_running_stats": True}
        nonlin_kwargs = {"inplace": True}
        conv_kwargs = {"kernel_size": 3, "stride": conv_stride, "padding": 1,
                       "dilation": 1, "bias": True}
        self.bnreluconv = nn.Sequential(*[nn.BatchNorm2d(input_feature_channels,
                                                         **norm_op_kwargs),
                                        nn.ReLU(**nonlin_kwargs),
                                        nn.Conv2d(input_feature_channels,
                                                  output_feature_channels,
                                                  **conv_kwargs)])
    def forward(self, x):
        return self.bnreluconv(x)

class PreActResidualBlock(nn.Module):
    """
    Pre-Activation Residual Block with convolutional downsampling support.
    """
    def __init__(self, input_feature_channels, output_feature_channels,
                 downsampling=True, bottleneck=True):

        super(PreActResidualBlock, self).__init__()
        first_conv_stride = 2 if downsampling else 1
        self.bnreluconv1 = BNReLUConv2d(input_feature_channels,
                                        output_feature_channels,
                                        conv_stride=first_conv_stride)
        self.bnreluconv2 = BNReLUConv2d(output_feature_channels,
                                        output_feature_channels,
                                        conv_stride=1)
        self.conv1x1 = nn.Conv2d(input_feature_channels,
                                 output_feature_channels,
                                 kernel_size=1, stride=first_conv_stride,
                                 padding=0)
        self.bottleneck = bottleneck
        self.output_channels = output_feature_channels

    def forward(self, x):
        out = self.bnreluconv1(x)
        out = self.bnreluconv2(out)
        identity = self.conv1x1(x) if self.bottleneck else x
        identity += out
        return identity

class UpsamplingBlock(nn.Module):
    """
    Transposed Conv2d + Pre-Activation Residual Block
    """
    def __init__(self, input_feature_channels, output_feature_channels):

        super(UpsamplingBlock, self).__init__()
        self.conv_up = nn.ConvTranspose2d(input_feature_channels,
                                          input_feature_channels,
                                          kernel_size=2, stride=2, padding=0,
                                          bias=False)
        # note: output_feature_channels = number of channels in skip connection
        skip_and_input_channels = output_feature_channels+input_feature_channels
        self.resblock = PreActResidualBlock(skip_and_input_channels,
                                            output_feature_channels,
                                            downsampling=False,
                                            bottleneck=True)
        self.output_channels = output_feature_channels

    def forward(self, x, skip_layer):
        upsampled = self.conv_up(x)
        skip_concat = torch.cat([upsampled, skip_layer], dim=1)
        res_output = self.resblock(skip_concat)
        return res_output
