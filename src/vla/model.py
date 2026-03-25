"""VLA model: VLM backbone + action query token + MLP action head."""

import torch
import torch.nn as nn

from vla.config import ACTION_DIM, CHUNK_SIZE


class VLA(nn.Module):
    def __init__(self, vlm, action_token_id: int, hidden_dim: int,
                 action_dim=ACTION_DIM, chunk_size=CHUNK_SIZE):
        super().__init__()
        self.vlm = vlm
        self.action_token_id = action_token_id
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, chunk_size * action_dim),
        )

    def forward(self, **vlm_inputs):
        outputs = self.vlm(**vlm_inputs, output_hidden_states=True)

        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)

        # Find <action> token position in each sequence
        action_mask = vlm_inputs["input_ids"] == self.action_token_id  # (B, seq_len)
        action_idx = action_mask.long().argmax(dim=1)  # (B,) — first occurrence
        action_hidden = last_hidden[torch.arange(last_hidden.shape[0]), action_idx]  # (B, hidden_dim)

        raw = self.action_head(action_hidden.float())  # (B, chunk_size * action_dim)
        return raw.view(-1, self.chunk_size, self.action_dim)  # (B, chunk_size, 7)
