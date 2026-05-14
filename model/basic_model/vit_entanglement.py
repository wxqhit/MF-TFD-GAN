import torch
import torch.nn as nn
from torch.nn import functional as F

# Vision Transformer (ViT) for feature extraction
class ViTFeatureExtractor(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12):
        """
        Vision Transformer (ViT) feature extractor
        :param image_size: input image size
        :param patch_size: image patch size
        :param in_channels: number of input channels
        :param embed_dim: embedding dimension
        :param depth: number of Transformer blocks
        :param num_heads: number of heads in multi-head attention
        """
        super(ViTFeatureExtractor, self).__init__()
        # Split the image into patch_size x patch_size patches and map each patch to embed_dim.
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Classification token used to generate the final classification feature.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Positional embedding used to encode each patch location.
        self.pos_embed = nn.Parameter(torch.randn(1, (image_size//patch_size)**2 + 1, embed_dim))
        # Transformer encoder blocks.
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=0.1)
            for _ in range(depth)
        ])
        # Layer normalization.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Forward pass
        :param x: input image with shape (B, C, H, W)
        :return: extracted features with shape (B, num_patches + 1, embed_dim)
        """
        B, C, H, W = x.shape
        # Split the image into patches and map them to embed_dim.
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        # Add the classification token.
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Add positional embedding.
        x = x+self.pos_embed
        # Pass through Transformer encoder blocks.
        for blk in self.blocks:
            x = blk(x)
        # Layer normalization.
        x = self.norm(x)
        return x

# Disentanglement module using ViT
class ViTDisentanglementModule(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        """
        ViT-based disentanglement module
        :param embed_dim: embedding dimension
        :param num_heads: number of heads in multi-head attention
        """
        super(ViTDisentanglementModule, self).__init__()
        # Shared Transformer encoder.
        self.shared_transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, dropout=0.1)
        # Linear layers for feature transformation.
        self.fc_r = nn.Linear(embed_dim, embed_dim)
        self.fc_u = nn.Linear(embed_dim, embed_dim)
        # Scaling factor.
        self.scale = embed_dim ** -0.5
        # Softmax function.
        self.softmax = nn.Softmax(dim=-1)
        # Log-softmax function.
        self.lsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, r, u):
        """
        Forward pass
        :param r: features of the first modality with shape (B, num_patches + 1, embed_dim)
        :param u: features of the second modality with shape (B, num_patches + 1, embed_dim)
        :return: disentangled features including r_public, r_private, u_public, and u_private
        """
        # Shared Transformer encoder.
        r = self.shared_transformer(r)
        u = self.shared_transformer(u)

        # Feature extraction.
        V_r = r
        V_u = u

        # Dual-output attention for r.
        Q_r = self.fc_r(r)
        K_r = self.fc_r(r)
        # Compute attention weights.
        A_rr = self.softmax(torch.matmul(Q_r, K_r.transpose(-2, -1)) * self.scale)
        A_ru = self.softmax(torch.matmul(Q_r, V_u.transpose(-2, -1)) * self.scale)
        # Compute public and private features.
        r_public = torch.matmul(A_rr, V_r)
        r_private = torch.matmul(A_ru, V_u)

        # Dual-output attention for u.
        Q_u = self.fc_u(u)
        K_u = self.fc_u(u)
        # Compute attention weights.
        A_ur = self.softmax(torch.matmul(Q_u, V_r.transpose(-2, -1)) * self.scale)
        A_uu = self.softmax(torch.matmul(Q_u, K_u.transpose(-2, -1)) * self.scale)
        # Compute public and private features.
        u_public = torch.matmul(A_ur, V_r)
        u_private = torch.matmul(A_uu, V_u)

        return r_public, r_private, u_public, u_private

# Complete model
class MultiModalViTDisentangler(nn.Module):
    def __init__(self, image_size=224, patch_size=16, embed_dim=768, num_heads=8):
        """
        Multimodal ViT disentanglement model
        :param image_size: input image size
        :param patch_size: image patch size
        :param embed_dim: embedding dimension
        :param num_heads: number of heads in multi-head attention
        """
        super(MultiModalViTDisentangler, self).__init__()
        # ViT feature extractor for the r modality.
        self.vit_r = ViTFeatureExtractor(image_size, patch_size, embed_dim=embed_dim)
        # ViT feature extractor for the u modality.
        self.vit_u = ViTFeatureExtractor(image_size, patch_size, embed_dim=embed_dim)
        # Disentanglement module.
        self.disentangler = ViTDisentanglementModule(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, r_image, u_image):
        """
        Forward pass
        :param r_image: image of the first modality with shape (B, C, H, W)
        :param u_image: image of the second modality with shape (B, C, H, W)
        :return: dictionary of disentangled features
        """
        # Feature extraction.
        r_features = self.vit_r(r_image)
        u_features = self.vit_u(u_image)

        # Disentanglement.
        r_public, r_private, u_public, u_private = self.disentangler(r_features, u_features)

        return {
            'r_public': r_public,
            'r_private': r_private,
            'u_public': u_public,
            'u_private': u_private
        }

# Example usage
if __name__ == "__main__":
    # Create model
    model = MultiModalViTDisentangler()

    # Create random input images
    r_image = torch.randn(1, 3, 224, 224)
    u_image = torch.randn(1, 3, 224, 224)

    # Forward pass
    output = model(r_image, u_image)

    # Print output shapes
    for key, value in output.items():
        print(f"{key}: {value.shape}")
