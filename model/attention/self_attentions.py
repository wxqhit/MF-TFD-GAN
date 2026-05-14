import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Dual_out_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super(Dual_out_Attention, self).__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head == -0.5
        self.norm_o = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.attend_o = nn.Softmax(dim=-1)
        self.attend_b = nn.Softmax(dim=-1)
        self.to_qkv_o = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_b = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out_1_o = nn.Linear(inner_dim, dim, bias=False)
        self.to_out_2_o = nn.Linear(inner_dim, dim, bias=False)
        self.to_out_1_b = nn.Linear(inner_dim, dim, bias=False)
        self.to_out_2_b = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x_o, x_b):
        x_o = self.norm_o(x_o)
        x_b = self.norm_b(x_b)
        qkv_o = self.to_qkv_o(x_o).chunk(3, dim=-1)
        q_o, k_o, v_o = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_o)
        qkv_b = self.to_qkv_o(x_b).chunk(3, dim=-1)
        q_b, k_b, v_b = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_b)

        dots_o2b = torch.matmul(q_b, k_o.transpose(-1, -2)) * self.scale
        dots_b2o = torch.matmul(q_o, k_b.transpose(-1, -2)) * self.scale
        attn_o = self.attend_o(dots_o2b)
        attn_b = self.attend_b(dots_b2o)
        out_o2b = torch.matmul(attn_o, v_o)
        out_o = torch.matmul(1 - attn_o, v_o)
        out_b2o = torch.matmul(attn_b, v_b)
        out_b = torch.matmul(1 - attn_b, v_b)
        out_o2b = rearrange(out_o2b, 'b h n d -> b n (h d)')
        out_o = rearrange(out_o, 'b h n d -> b n (h d)')
        out_b2o = rearrange(out_b2o, 'b h n d -> b n (h d)')
        out_b = rearrange(out_b, 'b h n d -> b n (h d)')

        out_o2b = self.to_out_1_o(out_o2b)
        out_o = self.to_out_2_o(out_o)
        out_b2o = self.to_out_1_b(out_b2o)
        out_b = self.to_out_2_b(out_b)
        return out_o2b, out_b2o, out_o, out_b


class Self_Attention(nn.Module):
    def __init__(self, dim, mlp_dim, heads=8, dim_head=64):
        super(Self_Attention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head == -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.ff = FeedForward(dim, mlp_dim)

    def forward(self, x):
        x_n = self.norm(x)
        qkv = self.to_qkv(x_n).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        x = self.to_out(out) + x
        x = self.ff(x) + x
        return x


class embedding(nn.Module):
    def __init__(self, channel, image_height, image_width, patch_height, patch_width, dim):
        super(embedding, self).__init__()
        patch_dim = channel * patch_width * patch_height
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding.to('cuda', dtype=x.dtype)
        return x
