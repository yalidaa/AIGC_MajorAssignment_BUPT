import math

import torch
import torch.nn as nn


try:
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
except Exception:
    OfficialMamba = None


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=28, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class LiteMambaBlock(nn.Module):
    """A lightweight Mamba-style fallback block for small vision assignments."""

    def __init__(self, dim, state_dim=16, expand=2, conv_kernel=3, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expand
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=hidden_dim,
        )
        self.state_gate = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)

        x_conv = self.depthwise_conv(x.transpose(1, 2))
        x_conv = x_conv[:, :, : x.size(1)].transpose(1, 2)
        x_conv = torch.nn.functional.silu(x_conv)

        selective_gate = self.state_gate(x_conv)
        x = x_conv * selective_gate * torch.nn.functional.silu(gate)
        x = self.out_proj(x)
        return residual + self.dropout(x)


class OfficialMambaBlock(nn.Module):
    def __init__(self, dim, state_dim=16, conv_kernel=4, expand=2, dropout=0.1):
        super().__init__()
        if OfficialMamba is None:
            raise ImportError("mamba-ssm is not installed")

        self.norm = nn.LayerNorm(dim)
        self.mamba = OfficialMamba(
            d_model=dim,
            d_state=state_dim,
            d_conv=conv_kernel,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class VisionMambaClassifier(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=4,
        embed_dim=64,
        depth=3,
        num_classes=2,
        state_dim=16,
        dropout=0.1,
        use_official=True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )

        block_cls = OfficialMambaBlock if use_official and OfficialMamba else LiteMambaBlock
        self.backend = "official-mamba-ssm" if block_cls is OfficialMambaBlock else "lite-mamba"
        self.blocks = nn.Sequential(
            *[
                block_cls(
                    dim=embed_dim,
                    state_dim=state_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([x, cls], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x[:, -1])
        return self.head(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class VisionTransformerClassifier(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=4,
        embed_dim=64,
        depth=3,
        num_heads=4,
        num_classes=2,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.backend = "transformer"

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([x, cls], dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x[:, -1])
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_attention_complexity(num_tokens):
    return int(math.pow(num_tokens, 2))
