import torch
import torch.nn as nn
from model.basic_model.spade_model import SPADE


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self, in_channel, out_channel, stride=1, downsample=None
    ):  # downsample��Ӧ���߲в�ṹ
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN����
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # �ݾ��ϵ����ֵ
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


# 50,101,152
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            bias=False,
        )  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            bias=False,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel * self.expansion,  # ���*4
            kernel_size=1,
            stride=1,
            bias=False,
        )  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.relu(out)

        return out


class SPABlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1):
        super(SPABlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.norm1 = SPADE(ks=3, norm_nc=out_channel, label_nc=out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               bias=False,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.norm2 = SPADE(ks=3, norm_nc=out_channel, label_nc=out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel * self.expansion,  # ���*4
                               kernel_size=1,
                               stride=1,
                               bias=False)  # unsqueeze channels
        self.norm3 = SPADE(ks=3, norm_nc=out_channel * self.expansion, label_nc=out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=in_channel * self.expansion,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(in_channel * self.expansion))

    def forward(self, x, seg):
        y = self.conv1(x)
        y = self.norm1(y, seg)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y, seg)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.norm3(y, seg)
        y = self.relu(y)

        y = y + self.downsample(x)
        return y

class down_sample(nn.Module):
    def __init__(self, in_channel, block_number):
        super(down_sample, self).__init__()
        self.in_channel = in_channel
        self.layer = self._make_layer(Bottleneck, in_channel, block_number, stride=2)

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channel,
                    channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channel * block.expansion),
            )

        layers = []
        layers.append(
            block(self.in_channel, channel, downsample=downsample, stride=stride)
        )
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        return x


class up_sample(nn.Module):
    def __init__(self, in_channel, cross_connect=True):
        super(up_sample, self).__init__()
        self.in_channel = in_channel
        self.cross_connect = cross_connect
        if cross_connect:
            channels = in_channel * 2
        else:
            channels = in_channel
        self.up_block = nn.Sequential(
            nn.ConvTranspose2d(channels, in_channel, kernel_size=2, stride=2),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        if self.cross_connect:
            y = self.up_block(torch.cat([x1, x2], dim=1))
        else:
            y = self.up_block(x1)
        return y
