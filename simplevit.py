import torch
import torch.nn as nn
from torch.nn import functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W] -> [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

    def forward(self, x):
        B = x.size(0)
        pos_embedding = self.pos_embed.repeat(B, 1, 1)
        return pos_embedding


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers, mlp_ratio, dropout):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x)


class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.classifier(x[:, 0])


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        n_heads,
        n_layers,
        mlp_ratio,
        n_classes,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEmbedding(self.patch_embed.n_patches, embed_dim)
        self.transformer_encoder = TransformerEncoder(
            embed_dim, n_heads, n_layers, mlp_ratio, dropout
        )
        self.classification_head = ClassificationHead(embed_dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed(x)
        x = self.transformer_encoder(x)
        x = self.classification_head(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, input_dim, num_experts, expert_dim, output_dim):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [nn.Linear(input_dim, expert_dim) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(input_dim, num_experts)
        self.output_proj = nn.Linear(expert_dim, output_dim)

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        outputs = torch.sum(gate_scores.unsqueeze(2) * outputs, dim=1)
        return self.output_proj(outputs)


class VisionTransformerWithMoE(VisionTransformer):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        n_heads,
        n_layers,
        mlp_ratio,
        n_classes,
        dropout=0.1,
        num_experts=4,
        expert_dim=2048,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            n_heads,
            n_layers,
            mlp_ratio,
            n_classes,
            dropout,
        )
        self.moe_layer = MoELayer(embed_dim, num_experts, expert_dim, embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed(x)
        x = self.moe_layer(x)
        x = self.transformer_encoder(x)
        x = self.classification_head(x)
        return x
