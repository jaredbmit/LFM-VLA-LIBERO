"""Sanity check: pass one batch through the VLM and inspect last-token hidden states."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from vla.data import CALVINDataset, make_calvin_collate_fn

CALVIN_BASE = "/home/jared/drl/calvin/dataset/calvin_debug_dataset"
MODEL_ID = "LiquidAI/LFM2-VL-3B"
SYSTEM_PROMPT = (
    "You are a robot manipulation agent. Given an image of the current scene "
    "and a language instruction, predict the next action to execute."
)
BATCH_SIZE = 4


def main():
    print(f"Loading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_image_tokens=256)
    processor.tokenizer.padding_side = "right"

    print(f"Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16,
    )
    model.eval()

    ds = CALVINDataset(f"{CALVIN_BASE}/training")
    collate_fn = make_calvin_collate_fn(processor, SYSTEM_PROMPT)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(loader))
    gt_actions = batch.pop("gt_actions")

    # Move to model device
    batch = {k: v.to(model.device) for k, v in batch.items()}

    print(f"\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True)

    # Last hidden state from the last layer
    last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden_dim)

    # Extract hidden state at the last non-padding token for each sequence
    seq_lengths = batch["attention_mask"].sum(dim=1)  # (B,)
    last_token_hidden = torch.stack([
        last_hidden[i, seq_lengths[i] - 1] for i in range(BATCH_SIZE)
    ])  # (B, hidden_dim)

    print(f"\nResults:")
    print(f"  last_hidden (full):    shape={list(last_hidden.shape)} dtype={last_hidden.dtype}")
    print(f"  last_token_hidden:     shape={list(last_token_hidden.shape)} dtype={last_token_hidden.dtype}")

    h = last_token_hidden.float()
    print(f"  mean={h.mean():.6f}  std={h.std():.6f}  min={h.min():.4f}  max={h.max():.4f}")
    print(f"  has NaN: {h.isnan().any().item()}  has Inf: {h.isinf().any().item()}")
    print(f"  norm per sample: {h.norm(dim=-1).tolist()}")

    print(f"\n  gt_actions shape: {list(gt_actions.shape)}")
    print(f"  gt_actions[0]: {gt_actions[0].tolist()}")

    # Check that different samples produce different hidden states
    cos_sim = torch.nn.functional.cosine_similarity(h[0].unsqueeze(0), h[1:], dim=-1)
    print(f"\n  cosine sim (sample 0 vs rest): {cos_sim.tolist()}")
    print(f"  (values < 1.0 confirm different inputs produce different representations)")


if __name__ == "__main__":
    main()
