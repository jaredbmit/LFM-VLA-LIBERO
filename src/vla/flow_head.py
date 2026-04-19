"""Conditional flow matching denoiser for VLA action prediction.

Architecture: small transformer that cross-attends to VLM hidden states,
conditioned on a diffusion timestep t ∈ [0, 1] via AdaLN-Zero modulation.
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for scalar timesteps t ∈ [0, 1] → (B, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
        )
        emb = t[:, None] * freqs[None, :]          # (B, half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return emb


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: (1 + scale) * x + shift. scale/shift are (B, d_model)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FlowTransformerLayer(nn.Module):
    """Pre-norm transformer block with AdaLN-Zero conditioning.

    Each sublayer (self-attn, cross-attn, FFN) gets its own (scale, shift, gate)
    produced from the timestep embedding. Gates are zero-initialized so each
    block starts as the identity and learns conditioning gradually.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # LayerNorms without learned affine — AdaLN supplies scale/shift
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

        # AdaLN-Zero: produces (scale, shift, gate) × 3 sublayers = 9 * d_model
        self.adaln_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, t_emb: torch.Tensor,
                ctx_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # t_emb: (B, d_model) → 9 modulation vectors of shape (B, d_model)
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.adaln_mod(t_emb).chunk(9, dim=-1)

        # Self-attention over action tokens
        h = _modulate(self.norm1(x), s1, b1)
        h, _ = self.self_attn(h, h, h)
        x = x + g1.unsqueeze(1) * self.drop(h)

        # Cross-attention to VLM context
        h = _modulate(self.norm2(x), s2, b2)
        h, _ = self.cross_attn(h, ctx, ctx, key_padding_mask=ctx_key_padding_mask)
        x = x + g2.unsqueeze(1) * self.drop(h)

        # FFN
        h = _modulate(self.norm3(x), s3, b3)
        x = x + g3.unsqueeze(1) * self.drop(self.ffn(h))
        return x


class FlowMatchingHead(nn.Module):
    """Flow matching denoiser: predicts velocity from noisy actions + timestep + VLM context.

    Args:
        action_dim: dimension of each action step (e.g. 7)
        chunk_size: number of action steps predicted jointly (e.g. 10)
        vlm_hidden_dim: hidden dimension of the VLM backbone
        d_model: internal transformer width
        n_heads: number of attention heads
        n_layers: number of transformer layers
        d_ff: FFN hidden dimension (defaults to 4 * d_model)
    """

    def __init__(self, action_dim: int, chunk_size: int, vlm_hidden_dim: int,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 4,
                 d_ff: int = 1024):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Action token embeddings
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Embedding(chunk_size, d_model)

        # Time conditioning: sinusoidal → MLP → d_model
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        # Project VLM hidden states to d_model
        self.context_proj = nn.Linear(vlm_hidden_dim, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            FlowTransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Final AdaLN-Zero before output projection (scale + shift only, no gate)
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        self.out_proj = nn.Linear(d_model, action_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        # AdaLN-Zero: zero the final linear of each modulation MLP so each
        # block starts as the identity (scale=0, shift=0, gate=0).
        for layer in self.layers:
            nn.init.zeros_(layer.adaln_mod[-1].weight)
            nn.init.zeros_(layer.adaln_mod[-1].bias)
        nn.init.zeros_(self.out_adaln[-1].weight)
        nn.init.zeros_(self.out_adaln[-1].bias)
        # DiT convention: zero the output projection too — the head starts
        # predicting zero velocity and learns from there.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, noisy_actions: torch.Tensor, t: torch.Tensor,
                vlm_context: torch.Tensor,
                context_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Predict velocity field at (x_t, t) conditioned on VLM context.

        Args:
            noisy_actions: (B, chunk_size, action_dim)
            t: (B,) timesteps in [0, 1]
            vlm_context: (B, seq_len, vlm_hidden_dim)
            context_key_padding_mask: (B, seq_len) bool, True = ignore (padding)

        Returns:
            velocity: (B, chunk_size, action_dim)
        """
        T = noisy_actions.shape[1]
        pos = torch.arange(T, device=noisy_actions.device)

        # Action tokens: projection + positional embedding
        x = self.action_proj(noisy_actions) + self.pos_embed(pos)[None, :, :]  # (B, T, d_model)

        # Timestep embedding — routed via AdaLN, not added to the residual stream
        t_emb = self.time_mlp(t)                            # (B, d_model)

        # Project VLM context
        ctx = self.context_proj(vlm_context)                # (B, seq_len, d_model)

        for layer in self.layers:
            x = layer(x, ctx, t_emb, context_key_padding_mask)

        # Final modulation + projection
        s, b = self.out_adaln(t_emb).chunk(2, dim=-1)
        return self.out_proj(_modulate(self.out_norm(x), s, b))
