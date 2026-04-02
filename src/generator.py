"""
Content-Style Decoupled Generator V3
=====================================
原始空间生成版本（V2.1 baseline）

Pipeline:
  src_h → NoiseGate → Content Encoder → Z_c
  src_h → Stats → Meta-Net → Δ
  Z_s = α·P_static + Δ
  Cross-Attn(Q=Z_c, KV=Z_s) → FFN → output_proj → raw features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class ContentEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2, dropout=dropout,
            norm_first=True, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.encoder(x))


class HybridStyleProvider(nn.Module):
    def __init__(self, embed_dim, num_style_tokens=4):
        super().__init__()
        self.num_style_tokens = num_style_tokens
        self.embed_dim = embed_dim

        self.static_prompts = nn.Parameter(
            torch.randn(num_style_tokens, embed_dim) * 0.02)

        self.meta_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, num_style_tokens * embed_dim))

        self.norm = nn.LayerNorm(embed_dim)
        self.mix_alpha = nn.Parameter(torch.tensor(0.5))
        self._init_weights()

    def _init_weights(self):
        for m in self.meta_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, src_features):
        batch_size = src_features.size(1)
        x_mean = src_features.mean(dim=0)
        x_var = src_features.var(dim=0).clamp(min=1e-6)
        stats = torch.cat([x_mean, x_var], dim=-1)

        dynamic_bias = self.meta_net(stats)
        dynamic_bias = dynamic_bias.view(batch_size, self.num_style_tokens, self.embed_dim)
        dynamic_bias = dynamic_bias.permute(1, 0, 2)

        static = self.static_prompts.unsqueeze(1).expand(-1, batch_size, -1)
        alpha = torch.sigmoid(self.mix_alpha)
        z_s = alpha * static + dynamic_bias

        return self.norm(z_s)


class ModalityGenerator(nn.Module):
    """原始空间生成器：输出 raw features，需要重编码"""

    def __init__(self, embed_dim, target_dims, num_heads=4, num_layers=2,
                 num_style_tokens=4, dropout=0.1):
        super().__init__()

        self.noise_gate = NoiseGate(embed_dim)
        self.content_encoder = ContentEncoder(embed_dim, num_heads, num_layers, dropout)
        self.style_provider = HybridStyleProvider(embed_dim, num_style_tokens)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=False)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim))
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn_dropout = nn.Dropout(dropout)

        # 各模态的输出投影（embed_dim → raw_dim）
        self.output_projs = nn.ModuleDict()
        for mod, dim in target_dims.items():
            self.output_projs[mod] = nn.Linear(embed_dim, dim)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if any(k in name for k in ['content_encoder', 'style_provider', 'noise_gate']):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, src_features, target_modality, target_seq_len=None):
        """
        Returns:
            generated: (batch, target_seq_len, target_dim) — 原始空间
            z_c, z_s: 用于正交约束
        """
        src_clean = self.noise_gate(src_features)
        z_c = self.content_encoder(src_clean)
        z_s = self.style_provider(src_clean)

        residual = z_c
        attn_out, _ = self.cross_attn(query=z_c, key=z_s, value=z_s)
        fused = self.cross_attn_norm(residual + self.cross_attn_dropout(attn_out))

        residual = fused
        fused = self.ffn_norm(residual + self.ffn_dropout(self.ffn(fused)))

        if target_seq_len is not None and target_seq_len != fused.size(0):
            fused_t = fused.permute(1, 2, 0)
            fused_t = F.interpolate(fused_t, size=target_seq_len, mode='linear', align_corners=False)
            fused = fused_t.permute(2, 0, 1)

        fused = fused.transpose(0, 1)  # (batch, seq, dim)
        generated = self.output_projs[target_modality](fused)

        return generated, z_c, z_s


def orthogonality_loss(z_c, z_s):
    c_global = z_c.mean(dim=0)
    s_global = z_s.mean(dim=0)
    cos_sim = F.cosine_similarity(c_global, s_global, dim=-1)
    return cos_sim.abs().mean()