import torch
from torch import nn

from model.attention.self_attentions import embedding, Self_Attention, Dual_out_Attention


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class feature_path(nn.Module):
    def __init__(self, channel=1, image_height=256, image_width=256, patch_height=32, patch_width=32,
                 dim=256, mlp_dim=512):
        super(feature_path, self).__init__()

        self.embedding = embedding(channel=channel, image_height=image_height, image_width=image_width,
                                   patch_height=patch_height, patch_width=patch_width,
                                   dim=dim)
        self.self_attention1 = Self_Attention(dim=dim, mlp_dim=mlp_dim)
        self.self_attention2 = Self_Attention(dim=dim, mlp_dim=mlp_dim)
        self.self_attention3 = Self_Attention(dim=dim, mlp_dim=mlp_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.self_attention1(x)
        x = self.self_attention2(x)
        x = self.self_attention3(x)
        return x


class discriminator(nn.Module):
    def __init__(self, dim=256, num_class=2):
        super(discriminator, self).__init__()
        self.ori_path = feature_path(dim=dim)
        self.bin_path = feature_path(dim=dim)
        self.Dual_attn = Dual_out_Attention(dim=dim)
        self.sigmoid = nn.Sigmoid()

        self.head_o = nn.Linear(dim, num_class)
        self.head_b = nn.Linear(dim, num_class)
        self.head_c = nn.Linear(dim, num_class)
        self.head_fin = nn.Linear(num_class*3, 1)

    def forward(self, x_o, x_b):
        y_o = self.ori_path(x_o)
        y_b = self.bin_path(x_b)
        y_o2b, y_b2o, y_o, y_b = self.Dual_attn(y_o, y_b)
        y_com = (y_b2o + y_o2b) / 2
        judge_o = y_o.mean(dim=1)
        judge_b = y_b.mean(dim=1)
        judge_c = y_com.mean(dim=1)
        class_o = self.head_o(judge_o)
        class_o = self.sigmoid(class_o)
        class_b = self.head_b(judge_b)
        class_b = self.sigmoid(class_b)
        class_c = self.head_c(judge_c)
        class_c = self.sigmoid(class_c)
        out_class = self.head_fin(torch.cat([class_o, class_b, class_c], dim=-1))
        out_class = self.sigmoid(out_class)
        return out_class, y_o, y_b, y_o2b, y_b2o

class discriminator_t(nn.Module):
    def __init__(self, dim=256, num_class=2):
        super(discriminator_t, self).__init__()
        self.ori_path = feature_path(dim=dim)
        self.sigmoid = nn.Sigmoid()

        self.head_o = nn.Linear(dim, num_class)

    def forward(self, x_o):
        y_o = self.ori_path(x_o)
        judge_o = y_o.mean(dim=1)
        class_o = self.head_o(judge_o)
        class_o = self.sigmoid(class_o)
        return class_o

