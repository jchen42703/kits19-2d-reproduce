#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from torch import nn
import numpy as np

from .neural_network import SegmentationNetwork
from .utils import UpsamplingBlock, PreActResidualBlock

class ResUNet(SegmentationNetwork):
    """
    Recursive U-Net with residual connections. Flexible. Code is modeleded
    after the Generic_UNet in nnU-Net by Isensee et al.
    """
    def __init__(self, input_channels, base_num_features, num_classes,
                 num_pool, max_num_features=256):
        """
        Attributes:
            input_channels:
            base_num_features (int):
            num_classes (int): Going to be 3 for this challenge (softmax)
                2 if sigmoid
            num_pool (int): Number of downsampling operations (pool size=2)
                Original paper does 4 + (1 in the center block)
            max_num_features (int): Number of features to cap to.
                Defaults to 256.
        """
        super(ResUNet, self).__init__()

        self.num_classes = num_classes
        self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []

        self.first_conv = nn.Conv2d(input_channels, base_num_features//2,
                                    kernel_size=3, padding=1)
        # Setting up the context pathway (downsampling)
        output_features = base_num_features
        input_features = self.first_conv.out_channels

        for d in range(num_pool):
            # add convolutions
            downsampling = True if d > 0 else False
            block = PreActResidualBlock(input_features, output_features,
                                        downsampling=downsampling,
                                        bottleneck=True)
            self.conv_blocks_context.append(block)

            input_features = output_features
            output_features = int(np.round(output_features * 2))

            output_features = min(output_features, self.max_num_features)


        # now the bottleneck.
        # determine the first stride
        final_num_features = output_features
        self.conv_blocks_context.append(PreActResidualBlock(input_features,
                                                            final_num_features))

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            # self.conv_blocks_context[-1] is bottleneck, so start with -2
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].out_channels
            self.conv_blocks_localization.append(UpsamplingBlock(nfeatures_from_down,
                                                                 nfeatures_from_skip))
            final_num_features = nfeatures_from_skip

        # register all modules properly
        self.seg_output = nn.Conv2d(self.conv_blocks_localization[-1].out_channels,
                                    num_classes, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)

    def forward(self, x):
        skips = []
        seg_outputs = []
        x = self.first_conv(x)
        # iterating through the # of context blocks except the center bottleneck
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.conv_blocks_localization)):
            # iterating through skip connections and localization blocks
            ## Basically, from bottom to top (ignoring the center block)
            x = self.conv_blocks_localization[u](x, skips[-(u+1)])

        return self.seg_output(x)

if __name__ == "__main__":
    model = ResUNet(input_channels=5, base_num_features=16, num_classes=3,
                    num_pool=4, max_num_features=256)
    # calculating # of parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total # of Params: {total}\nTrainable params: {trainable}")
