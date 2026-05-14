import torch
import torch.nn as nn
from model.basic_model.basic_model import down_sample, up_sample, SPABlock
from model.basic_model.spade_model import SPADE
from model.attention.FCANet import MultiSpectralAttentionLayer

class enconder_path(nn.Module):
    def __init__(self, in_channel, SPA_flag=True, high_freq=False):
        super(enconder_path, self).__init__()
        self.in_channel = in_channel
        self.high_freq = high_freq
        self.SPA_flag = SPA_flag  # use spanorm or not
        if self.high_freq:
            channel_init = 32
        else:
            channel_init = 64
        self.conv1 = nn.Conv2d(self.in_channel, channel_init,
                               kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(channel_init)
        self.relu = nn.ReLU(inplace=True)
        block_num = [3, 4, 6, 3]
        if self.SPA_flag:
            block_num = [2, 3, 5, 2]
            self.SPAblock1 = SPABlock(128, 128)
            self.SPAblock2 = SPABlock(256, 256)
            self.SPAblock3 = SPABlock(512, 512)
            self.SPAblock4 = SPABlock(1024, 1024)
        if self.high_freq:
            self.down_layer_high = down_sample(32, 3)  # out size 64
        self.down_layer1 = down_sample(64, block_number=block_num[0])  # out size 32
        self.down_layer2 = down_sample(128, block_number=block_num[1])  # out size 16
        self.down_layer3 = down_sample(256, block_number=block_num[2])  # out size 8
        self.down_layer4 = down_sample(512, block_number=block_num[3])  # out size 4


    def forward(self, x, seg1, seg2, seg3, seg4):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.high_freq:
            x = self.down_layer_high(x)
        x1 = self.down_layer1(x)
        if self.SPA_flag:
            x1 = self.SPAblock1(x1, seg1)
        x2 = self.down_layer2(x1)
        if self.SPA_flag:
            x2 = self.SPAblock2(x2, seg2)
        x3 = self.down_layer3(x2)
        if self.SPA_flag:
            x3 = self.SPAblock3(x3, seg3)
        x4 = self.down_layer4(x3)
        if self.SPA_flag:
            x4 = self.SPAblock4(x4, seg4)
        return x, x1, x2, x3, x4

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.path1 = enconder_path(in_channel=1, SPA_flag=False, high_freq=False)  # low frequency
        self.path2 = enconder_path(in_channel=3, SPA_flag=True, high_freq=False)  # middle frequency
        self.path3 = enconder_path(in_channel=3, SPA_flag=True, high_freq=True)  # high frequency

        self.cross_connect_0 = nn.Sequential(
            MultiSpectralAttentionLayer(64*3, 64, 64),
            nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=1, stride=1),  # 64*3
        )

        self.cross_connect_1 = nn.Sequential(
            MultiSpectralAttentionLayer(128*3, 32, 32),
            nn.Conv2d(in_channels=128 * 3, out_channels=128, kernel_size=1, stride=1),  # 128*3
        )
        self.cross_connect_2 = nn.Sequential(
            MultiSpectralAttentionLayer(256*3, 16, 16),
            nn.Conv2d(in_channels=256 * 3, out_channels=256, kernel_size=1, stride=1),  # 256*3
        )
        self.cross_connect_3 = nn.Sequential(
            MultiSpectralAttentionLayer(512*3, 8, 8),
            nn.Conv2d(in_channels=512 * 3, out_channels=512, kernel_size=1, stride=1),  # 512*3
        )

        self.feat_cat = nn.Conv2d(in_channels=1024 * 3, out_channels=1024, kernel_size=1, stride=1)

    def forward(self, x_l, x_m, x_h):
        y_l, y_l1, y_l2, y_l3, y_l4 = self.path1(x_l, x_l, x_l, x_l, x_l)
        y_m, y_m1, y_m2, y_m3, y_m4 = self.path2(x_m, y_l1, y_l2, y_l3, y_l4)
        y_h, y_h1, y_h2, y_h3, y_h4 = self.path3(x_h, y_m1, y_m2, y_m3, y_m4)

        y_cross_0 = self.cross_connect_0(torch.cat([y_l, y_m, y_h], dim=1))
        y_cross_1 = self.cross_connect_1(torch.cat([y_l1, y_m1, y_h1], dim=1))
        y_cross_2 = self.cross_connect_2(torch.cat([y_l2, y_m2, y_h2], dim=1))
        y_cross_3 = self.cross_connect_3(torch.cat([y_l3, y_m3, y_h3], dim=1))

        y = self.feat_cat(torch.cat([y_l4, y_m4, y_h4], dim=1))
        return y, y_cross_0, y_cross_1, y_cross_2, y_cross_3

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.up_layer0 = up_sample(in_channel=1024, cross_connect=False)
        self.up_layer1 = up_sample(in_channel=512, cross_connect=True)
        self.up_layer2 = up_sample(in_channel=256, cross_connect=True)
        self.up_layer3 = up_sample(in_channel=128, cross_connect=True)
        self.up_layer4 = up_sample(in_channel=64, cross_connect=True)

        self.up_layer5 = up_sample(in_channel=32, cross_connect=False)

        self.out_block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x, x_cross_0, x_cross_1, x_cross_2, x_cross_3):
        x = self.up_layer0(x, x)
        x = self.up_layer1(x, x_cross_3)
        x = self.up_layer2(x, x_cross_2)
        x = self.up_layer3(x, x_cross_1)
        x = self.up_layer4(x, x_cross_0)
        x = self.up_layer5(x, x)
        x = self.out_block(x)
        return x

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self, x_l, x_m, x_h):
        y, y_cross_0, y_cross_1, y_cross_2, y_cross_3 = self.encoder(x_l, x_m, x_h)
        out = self.decoder(y, y_cross_0, y_cross_1, y_cross_2, y_cross_3)
        return out


