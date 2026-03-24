"""Demo script for the CALVIN dataloader with action chunking and action query token."""

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from vla import ACTION_TOKEN
from vla.data import CALVINDataset, make_calvin_collate_fn

CALVIN_BASE = "/home/jared/drl/calvin/dataset/calvin_debug_dataset"
MODEL_ID = "LiquidAI/LFM2-VL-3B"
SYSTEM_PROMPT = (
    "You are a robot manipulation agent. Given an image of the current scene "
    "and a language instruction, predict the next action to execute."
)
BATCH_SIZE = 4
CHUNK_SIZE = 10


def main():
    # --- Datasets ---
    train_ds = CALVINDataset(f"{CALVIN_BASE}/training", chunk_size=CHUNK_SIZE)
    val_ds = CALVINDataset(f"{CALVIN_BASE}/validation", chunk_size=CHUNK_SIZE)
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    # Peek at one sample
    sample = train_ds[0]
    print(f"\nSample 0:")
    print(f"  image:        {sample['image'].size} mode={sample['image'].mode}")
    print(f"  instruction:  {sample['instruction']!r}")
    print(f"  action_chunk: shape={list(sample['action_chunk'].shape)} dtype={sample['action_chunk'].dtype}")
    print(f"  frame_id:     {sample['frame_id']}")

    # --- Processor + action token ---
    print(f"\nLoading processor from {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_image_tokens=256)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [ACTION_TOKEN]})
    action_token_id = processor.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)
    print(f"Action token: {ACTION_TOKEN!r} -> id={action_token_id}")

    collate_fn = make_calvin_collate_fn(processor, SYSTEM_PROMPT)

    # Use samples from different annotations to test padding behavior
    subset = Subset(train_ds, [0, 100, 200, 300])
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(loader))

    # --- Tensor shapes ---
    print("\n--- Batch shapes ---")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s} shape={str(list(v.shape)):15s} dtype={v.dtype}")

    # --- Action token positioning ---
    ids = batch["input_ids"]
    mask = batch["attention_mask"]

    print("\n--- Action token positioning ---")
    for i in range(ids.shape[0]):
        real_len = mask[i].sum().item()
        action_pos = (ids[i] == action_token_id).nonzero(as_tuple=True)[0].tolist()
        is_last = action_pos[0] == real_len - 1 if action_pos else False

        # Show last few real tokens
        start = max(0, real_len - 4)
        tokens_str = " ".join(
            repr(processor.tokenizer.decode([ids[i, j]])) for j in range(start, real_len)
        )
        print(f"  [{i}] real_len={real_len} action_at={action_pos} is_last_real={is_last}")
        print(f"       tail: {tokens_str}")

    # --- Decoded text ---
    text = processor.tokenizer.decode(ids[0], skip_special_tokens=False)
    print(f"\n--- Decoded text (sample 0, first 1000 chars) ---\n  {text[:1000]}")

    # --- Action chunk stats ---
    actions = batch["gt_actions"]
    print(f"\n--- Action chunks ---")
    print(f"  shape: {list(actions.shape)}")
    print(f"  pose (dims 0-5) range: [{actions[:,:,:6].min():.4f}, {actions[:,:,:6].max():.4f}]")
    print(f"  gripper (dim 6) values: {sorted(set(actions[:,:,6].flatten().tolist()))}")

    print("\nDone.")


if __name__ == "__main__":
    main()
