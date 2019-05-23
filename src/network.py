from torch import nn
from src.utils import initialize_weights
import torch

class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.PReLU()

        )
        self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

        self._initialize_weights()

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)


class FastARCNN(nn.Module):
    def __init__(self):
        super(FastARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.ConvTranspose2d(64, 3, kernel_size=9, stride=2, padding=4, output_padding=1)

        self._initialize_weights()

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)


class DnCNN(nn.Module):
    def __init__(self, depth=15, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(
            nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                          bias=True))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding,
                      bias=True))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y + out

    def _initialize_weights(self):
        for m in self.modules():
            initialize_weights(m)


class ESPCN(nn.Module):
#upscale_factor -> args
    def __init__(self):
        super(ESPCN, self).__init__()
#        print("Creating ESPCN (x%d)" %args.scale)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(x)
        x = self.conv4(x)
        return x


class RESCNN(nn.Module):
    def __init__(self):
        super(RESCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, 32, kernel_size=7, padding=3))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(32, 64, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 256, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        '''for _ in range(1):
            layers.append(nn.Conv2d(256, 64, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(64, 256, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))'''
        layers.append(nn.Conv2d(256, 64, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 512, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        '''for _ in range(2):
            layers.append(nn.Conv2d(512, 128, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(128, 512, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))'''
        layers.append(nn.Conv2d(512, 128, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, 1024, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        '''for _ in range(4):
            layers.append(nn.Conv2d(1024, 256, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(256, 1024, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))'''
        layers.append(nn.Conv2d(1024, 256, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(512, 2048, kernel_size=1, padding=0))
        layers.append(nn.ReLU(inplace=True))
        '''for _ in range(1):
            layers.append(nn.Conv2d(2048, 512, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(512, 2048, kernel_size=1, padding=0))
            layers.append(nn.ReLU(inplace=True))'''
        layers.append(nn.Conv2d(2048, 3, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        self.rescnn = nn.Sequential(*layers)


    def forward(self, x):
        out = self.rescnn(x)
        return out

    def _initialize_weights():
        for m in self.modules():
            initialize_weights(m)


class MemNet(nn.Module):
    def __init__(self, in_channels, channels, num_memblock, num_resblock):
        super(MemNet, self).__init__()
        self.feature_extractor = BNReLUConv(in_channels, channels)
        self.reconstructor = BNReLUConv(channels, in_channels)
        self.dense_memory = nn.ModuleList(
            [MemoryBlock(channels, num_resblock, i + 1) for i in range(num_memblock)]
        )
        self.final_layer = nn.Sequential(nn.Conv2d(in_channels, 3, kernel_size=3, padding=1), nn.ReLU(inplace=True))
    def forward(self, x):
        # x = x.contiguous()
        residual = x
        out = self.feature_extractor(x)
        ys = [out]
        for memory_block in self.dense_memory:
            out = memory_block(out, ys)
        out = self.reconstructor(out)
        out = out + residual
        out = self.final_layer(out)
        return out


class MemoryBlock(nn.Module):
    """Note: num_memblock denotes the number of MemoryBlock currently"""

    def __init__(self, channels, num_resblock, num_memblock):
        super(MemoryBlock, self).__init__()
        self.recursive_unit = nn.ModuleList(
            [ResidualBlock(channels) for i in range(num_resblock)]
        )
        self.gate_unit = BNReLUConv((num_resblock + num_memblock) * channels, channels, 1, 1, 0)

    def forward(self, x, ys):
        """ys is a list which contains long-term memory coming from previous memory block
        xs denotes the short-term memory coming from recursive unit
        """
        xs = []
        residual = x
        for layer in self.recursive_unit:
            x = layer(x)
            xs.append(x)

        gate_out = self.gate_unit(torch.cat(xs + ys, 1))
        ys.append(gate_out)
        return gate_out


class ResidualBlock(nn.Module):
    def __init__(self, channels, k=3, s=1, p=1):
        super(ResidualBlock, self).__init__()
        self.relu_conv1 = BNReLUConv(channels, channels, k, s, p)
        self.relu_conv2 = BNReLUConv(channels, channels, k, s, p)

    def forward(self, x):
        residual = x
        out = self.relu_conv1(x)
        out = self.relu_conv2(out)
        out = out + residual
        return out


class BNReLUConv(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(BNReLUConv, self).__init__()
        self.add_module('bn', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))

