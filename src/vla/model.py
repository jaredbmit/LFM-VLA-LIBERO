"""VLA model: VLM backbone + action query token + split action heads."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla.config import ACTION_DIM, ACTION_TOKEN, CHUNK_SIZE, GRIPPER_LOSS_WEIGHT
from vla.flow_head import FlowMatchingHead


def masked_action_mse(pred: torch.Tensor, gt: torch.Tensor,
                      action_mask: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and ground-truth actions, masked by action_mask."""
    mask = action_mask.unsqueeze(-1)
    return ((pred - gt) ** 2 * mask).sum() / mask.sum() / pred.shape[-1]


def _xavier_init(module: nn.Module):
    """Apply Xavier uniform initialization to all Linear layers in a module."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def install_action_token(tokenizer, vlm) -> int:
    """Register <action> in the tokenizer and resize the VLM embedding to match."""
    if ACTION_TOKEN not in tokenizer.get_added_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [ACTION_TOKEN]})
        vlm.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(ACTION_TOKEN)


class VLA(nn.Module):
    """MLP action head with a learnable <action> query token.

    The caller is responsible for having already called
    `install_action_token(tokenizer, vlm)` and passing the resulting id here.
    """

    def __init__(self, vlm, action_token_id: int, hidden_dim: int,
                 action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE):
        super().__init__()
        self.vlm = vlm
        self.action_token_id = action_token_id
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Learnable action query — replaces the embedding lookup at action-token
        # positions on every forward, so the underlying embed_tokens row is dead.
        self.action_query = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.action_query, std=0.02)
        self._register_action_query_hook()

        head_dim = 1024

        # Project VLM hidden state to shared action representation
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, head_dim),
        )

        # Pose head: 6D end-effector deltas, bounded by Tanh to [-1, 1]
        self.pose_head = nn.Sequential(
            nn.Linear(head_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, chunk_size * (action_dim - 1)),
            nn.Tanh(),
        )

        # Gripper head: binary logits (raw, for BCEWithLogitsLoss)
        self.gripper_head = nn.Sequential(
            nn.Linear(head_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, chunk_size),
        )

        _xavier_init(self.proj)
        _xavier_init(self.pose_head)
        _xavier_init(self.gripper_head)

    def _register_action_query_hook(self) -> None:
        """Splice self.action_query into the embedding output at <action> positions."""
        embed = self.vlm.get_input_embeddings()

        def hook(module, inputs, output):
            ids = inputs[0] if inputs else None
            if ids is None:
                return output
            mask = (ids == self.action_token_id)
            if not mask.any():
                return output
            output = output.clone()
            output[mask] = self.action_query.to(output.dtype)
            return output

        embed.register_forward_hook(hook)

    def forward(self, **vlm_inputs):
        outputs = self.vlm(**vlm_inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)

        # Find <action> token position in each sequence
        action_mask = vlm_inputs["input_ids"] == self.action_token_id  # (B, seq_len)
        action_idx = action_mask.long().argmax(dim=1)  # (B,) — first occurrence
        action_hidden = last_hidden[torch.arange(last_hidden.shape[0]), action_idx]  # (B, hidden_dim)

        h = self.proj(action_hidden.float())  # (B, head_dim)

        pose = self.pose_head(h).view(-1, self.chunk_size, self.action_dim - 1)  # (B, chunk, 6)
        gripper = self.gripper_head(h).view(-1, self.chunk_size, 1)  # (B, chunk, 1)

        return torch.cat([pose, gripper], dim=-1)  # (B, chunk_size, 7)

    @torch.no_grad()
    def predict_actions(self, **vlm_inputs) -> torch.Tensor:
        """Return actions with gripper mapped to [-1, 1] via sigmoid (matches eval-time postprocess)."""
        pred = self(**vlm_inputs)
        gripper = 2 * torch.sigmoid(pred[..., 6:7]) - 1
        return torch.cat([pred[..., :6], gripper], dim=-1)

    def compute_loss(self, vlm_inputs: dict, gt_actions: torch.Tensor,
                     action_mask: torch.Tensor) -> torch.Tensor:
        """Huber loss on pose dims + weighted BCE on gripper, masked by action_mask."""
        pred = self(**vlm_inputs)                   # (B, chunk, 7)
        mask = action_mask.unsqueeze(-1)            # (B, chunk, 1)

        pose_loss = F.smooth_l1_loss(pred[..., :6], gt_actions[..., :6], reduction="none")
        pose_loss = (pose_loss * mask).sum() / mask.sum() / 6

        gripper_target = (gt_actions[..., 6] + 1) / 2  # {-1, 1} -> {0, 1}
        gripper_loss = F.binary_cross_entropy_with_logits(
            pred[..., 6], gripper_target, reduction="none"
        )
        gripper_loss = (gripper_loss * action_mask).sum() / action_mask.sum()

        return pose_loss + GRIPPER_LOSS_WEIGHT * gripper_loss

    def head_state_dict(self) -> dict:
        """Serializable state for this head type (used by checkpointing)."""
        return {
            "head": "mlp",
            "action_query": self.action_query.detach().cpu(),
            "proj": self.proj.state_dict(),
            "pose_head": self.pose_head.state_dict(),
            "gripper_head": self.gripper_head.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, vlm, ckpt: dict, *, action_token_id: int,
                        hidden_dim: int, **_) -> "VLA":
        vla = cls(vlm, action_token_id=action_token_id, hidden_dim=hidden_dim)
        if "action_query" in ckpt:
            with torch.no_grad():
                vla.action_query.copy_(ckpt["action_query"].to(vla.action_query.device))
        vla.proj.load_state_dict(ckpt["proj"])
        vla.pose_head.load_state_dict(ckpt["pose_head"])
        vla.gripper_head.load_state_dict(ckpt["gripper_head"])
        return vla


def _num_vlm_layers(vlm) -> int:
    """Return the number of transformer blocks in a HuggingFace VLM."""
    cfg = vlm.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    # Some VLMs nest the LM config under `text_config`.
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
        return cfg.text_config.num_hidden_layers
    raise ValueError(f"Cannot determine num_hidden_layers from {type(cfg).__name__}")


class FlowMatchingVLA(nn.Module):
    """VLA with a conditional flow matching action head.

    The flow head cross-attends to an intermediate VLM layer (~70% depth) rather
    than the final layer, which tends to be biased toward next-token prediction.

    Training: call get_vlm_hidden() once, then forward_train() with the flow
    matching inputs (noisy action + timestep).
    Inference: call forward() or denoise() directly.
    """

    def __init__(self, vlm, hidden_dim: int,
                 action_dim: int = ACTION_DIM, chunk_size: int = CHUNK_SIZE,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 4,
                 n_steps_inference: int = 10):
        super().__init__()
        self.vlm = vlm
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_steps_inference = n_steps_inference

        # hidden_states[0] = embeddings, hidden_states[i] = output of block i-1.
        # We target 70% depth: e.g. for a 28-layer model → block 20 → index 20.
        n_vlm_layers = _num_vlm_layers(vlm)
        self.fusion_layer_idx = round(0.7 * n_vlm_layers)

        self.flow_head = FlowMatchingHead(
            action_dim=action_dim,
            chunk_size=chunk_size,
            vlm_hidden_dim=hidden_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        )

    def get_vlm_hidden(self, **vlm_inputs) -> torch.Tensor:
        """Run VLM and return hidden states at fusion_layer_idx (B, seq_len, hidden_dim)."""
        out = self.vlm(**vlm_inputs, output_hidden_states=True)
        return out.hidden_states[self.fusion_layer_idx]

    @torch.no_grad()
    def forward(self, *, n_steps: int | None = None, **vlm_inputs) -> torch.Tensor:
        """Inference: run VLM then Euler-integrate the velocity field from noise.

        Returns:
            actions (B, chunk_size, action_dim)
        """
        if n_steps is None:
            n_steps = self.n_steps_inference
        ctx_mask = None
        if "attention_mask" in vlm_inputs:
            ctx_mask = vlm_inputs["attention_mask"] == 0  # True = padding
        ctx = self.get_vlm_hidden(**vlm_inputs).float()

        B = ctx.shape[0]
        x = torch.randn(B, self.chunk_size, self.action_dim,
                        device=ctx.device, dtype=torch.float32)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=ctx.device, dtype=torch.float32)
            x = x + self.flow_head(x, t, ctx, ctx_mask) * dt
        return x

    @torch.no_grad()
    def predict_actions(self, **vlm_inputs) -> torch.Tensor:
        """Full denoising pass; returns actions in the same space as gt_actions."""
        return self(**vlm_inputs)

    def compute_loss(self, vlm_inputs: dict, gt_actions: torch.Tensor,
                     action_mask: torch.Tensor) -> torch.Tensor:
        """Conditional flow matching loss: masked MSE on predicted velocity."""
        ctx_mask = None
        if "attention_mask" in vlm_inputs:
            ctx_mask = vlm_inputs["attention_mask"] == 0
        vlm_hidden = self.get_vlm_hidden(**vlm_inputs)

        B = gt_actions.shape[0]
        x0 = torch.randn_like(gt_actions)
        t = torch.rand(B, device=gt_actions.device)
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * gt_actions
        v_target = gt_actions - x0

        v_pred = self.flow_head(x_t.float(), t, vlm_hidden.float(), ctx_mask)
        mask = action_mask.unsqueeze(-1)
        return ((v_pred - v_target) ** 2 * mask).sum() / mask.sum() / self.action_dim

    def head_state_dict(self) -> dict:
        """Serializable state for this head type (used by checkpointing)."""
        return {
            "head": "flow",
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "fusion_layer_idx": self.fusion_layer_idx,
            "flow_head": self.flow_head.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, vlm, ckpt: dict, *, hidden_dim: int,
                        n_steps_inference: int = 10, **_) -> "FlowMatchingVLA":
        vla = cls(
            vlm, hidden_dim=hidden_dim,
            d_model=ckpt["d_model"], n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
            n_steps_inference=n_steps_inference,
        )
        vla.flow_head.load_state_dict(ckpt["flow_head"])
        return vla
