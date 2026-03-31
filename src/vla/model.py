"""VLA model: VLM backbone + action query token + split action heads."""

import torch
import torch.nn as nn

from vla.config import ACTION_DIM, CHUNK_SIZE


def _xavier_init(module: nn.Module):
    """Apply Xavier uniform initialization to all Linear layers in a module."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class VLA(nn.Module):
    def __init__(self, vlm, action_token_id: int, hidden_dim: int,
                 action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE):
        super().__init__()
        self.vlm = vlm
        self.action_token_id = action_token_id
        self.chunk_size = chunk_size
        self.action_dim = action_dim

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
