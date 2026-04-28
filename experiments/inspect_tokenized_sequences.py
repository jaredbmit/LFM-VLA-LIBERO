"""Inspect tokenized sequences across all VLM backends.

Passes CALVIN samples through each model's processor/collate and prints the
full token sequence annotated with key markers: image tokens, special
delimiters, system/user/assistant roles, and the <action> token.
"""

import argparse

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor

from vla import ACTION_TOKEN, MODEL_REGISTRY, SYSTEM_PROMPT
from vla.config import CHUNK_SIZE
from vla.data import CALVINDataset, make_calvin_collate_fn

CALVIN_BASE = "/home/jared/drl/calvin/dataset/calvin_debug_dataset"


def inspect_model(model_name: str, dataset: CALVINDataset, sample_indices: list[int],
                  batch_size: int = 4):
    spec = MODEL_REGISTRY[model_name]
    print(f"\n{'=' * 80}")
    print(f"MODEL: {model_name}  ({spec.model_id})")
    print(f"  processor_kwargs = {spec.processor_kwargs}")
    print(f"{'=' * 80}")

    processor = AutoProcessor.from_pretrained(spec.model_id, **spec.processor_kwargs)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": [ACTION_TOKEN]})
    action_token_id = tok.convert_tokens_to_ids(ACTION_TOKEN)

    collate_fn = make_calvin_collate_fn(
        processor, SYSTEM_PROMPT,
        max_length=spec.max_length,
        action_token_id=action_token_id,
    )

    subset = Subset(dataset, sample_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    B, S = ids.shape

    # ---- Batch overview ----
    print(f"\n--- Batch tensor shapes ---")
    for k, v in sorted(batch.items()):
        if isinstance(v, torch.Tensor):
            print(f"  {k:30s} {str(list(v.shape)):20s} {v.dtype}")

    print(f"\n--- Vocabulary info ---")
    print(f"  vocab_size       = {tok.vocab_size}")
    print(f"  pad_token        = {tok.pad_token!r}  (id={tok.pad_token_id})")
    print(f"  bos_token        = {tok.bos_token!r}  (id={tok.bos_token_id})")
    print(f"  eos_token        = {tok.eos_token!r}  (id={tok.eos_token_id})")
    print(f"  action_token     = {ACTION_TOKEN!r}  (id={action_token_id})")

    # ---- Per-sample analysis ----
    for i in range(B):
        real_len = mask[i].sum().item()
        sample = dataset[sample_indices[i]]

        print(f"\n{'─' * 70}")
        print(f"  Sample {i}  (frame={sample['frame_id']}, instruction={sample['instruction']!r})")
        print(f"  Sequence length: {real_len} real tokens + {S - real_len} padding = {S} total")

        token_ids_all = ids[i].tolist()
        token_ids_real = token_ids_all[:real_len]
        pad_token_id = tok.pad_token_id

        image_token_ids = _get_image_token_ids(tok, model_name)
        n_image = sum(1 for t in token_ids_real if t in image_token_ids)
        print(f"  Image tokens: {n_image}  (ids: {sorted(image_token_ids)})")

        # Action token position
        action_positions = (ids[i] == action_token_id).nonzero(as_tuple=True)[0].tolist()
        print(f"  Action token positions: {action_positions}")
        if action_positions:
            is_last_real = action_positions[-1] == real_len - 1
            print(f"  Action is last real token: {is_last_real}")

        # ---- Structural summary (includes padding) ----
        print(f"\n  --- Sequence structure ---")
        _print_structure(tok, token_ids_all, real_len, image_token_ids,
                         action_token_id, pad_token_id)

        # ---- Token-by-token view of boundaries ----
        print(f"\n  --- Token-by-token (first 30 + around image boundaries + last 20) ---")
        _print_token_ranges(tok, token_ids_all, real_len, image_token_ids,
                            action_token_id, pad_token_id)

    print()


def _get_image_token_ids(tok, model_name: str) -> set[int]:
    """Find the token IDs used as image placeholders for each model."""
    candidates = set()
    # Try common image placeholder token names
    for name in ["<image>", "<|image_pad|>", "<|placeholder|>",
                 "<|image|>", "<image_token>", "<img>"]:
        tid = tok.convert_tokens_to_ids(name)
        # convert_tokens_to_ids returns unk_token_id if not found
        if tid != tok.unk_token_id:
            candidates.add(tid)

    # Qwen uses <|image_pad|> — also check the added_tokens
    if hasattr(tok, 'added_tokens_encoder'):
        for token_str, token_id in tok.added_tokens_encoder.items():
            if 'image' in token_str.lower() and 'pad' in token_str.lower():
                candidates.add(token_id)

    return candidates


def _print_structure(tok, token_ids: list[int], real_len: int,
                     image_token_ids: set[int], action_token_id: int,
                     pad_token_id: int):
    """Print a compact structural breakdown: collapse runs of image/pad/text tokens."""
    segments = []  # list of (label, start, end, [extra])
    i = 0
    n = len(token_ids)
    while i < n:
        tid = token_ids[i]
        if tid == pad_token_id and i >= real_len:
            # Consume run of padding tokens
            j = i
            while j < n and token_ids[j] == pad_token_id:
                j += 1
            segments.append(("PAD", i, j))
            i = j
        elif tid in image_token_ids:
            j = i
            while j < n and token_ids[j] in image_token_ids:
                j += 1
            segments.append(("IMAGE", i, j))
            i = j
        elif tid == action_token_id:
            segments.append(("ACTION", i, i + 1))
            i += 1
        else:
            j = i
            while (j < n and token_ids[j] not in image_token_ids
                   and token_ids[j] != action_token_id
                   and not (token_ids[j] == pad_token_id and j >= real_len)):
                j += 1
            text = tok.decode(token_ids[i:j], skip_special_tokens=False)
            text = text.replace('\n', '\\n')
            segments.append(("TEXT", i, j, text))
            i = j

    for seg in segments:
        if seg[0] == "IMAGE":
            _, start, end = seg
            print(f"    [{start:4d}..{end - 1:4d}]  IMAGE x{end - start}")
        elif seg[0] == "PAD":
            _, start, end = seg
            print(f"    [{start:4d}..{end - 1:4d}]  PAD x{end - start}")
        elif seg[0] == "ACTION":
            _, start, _ = seg
            print(f"    [{start:4d}      ]  <action>")
        else:
            _, start, end, text = seg
            display = text if len(text) <= 90 else text[:44] + " ... " + text[-44:]
            print(f"    [{start:4d}..{end - 1:4d}]  {display}")


def _print_token_ranges(tok, token_ids: list[int], real_len: int,
                        image_token_ids: set[int], action_token_id: int,
                        pad_token_id: int):
    """Print specific token ranges: head, around image boundaries, action, tail with padding."""
    n = len(token_ids)

    ranges = []
    # First 30
    ranges.append((0, min(30, n), "HEAD"))
    # Find image region boundaries
    in_image = False
    img_start = img_end = -1
    for j, tid in enumerate(token_ids[:real_len]):
        if tid in image_token_ids and not in_image:
            img_start = j
            in_image = True
        elif tid not in image_token_ids and in_image:
            img_end = j
            in_image = False
    if in_image:
        img_end = real_len
    if img_start >= 0:
        a, b = max(0, img_start - 3), min(n, img_start + 10)
        ranges.append((a, b, f"IMAGE START (pos {img_start})"))
        a, b = max(0, img_end - 5), min(n, img_end + 5)
        ranges.append((a, b, f"IMAGE END (pos {img_end})"))
    # Last 20 real tokens + padding boundary + first few pad tokens
    tail_start = max(0, real_len - 15)
    tail_end = min(n, real_len + 5)  # show a few padding tokens too
    ranges.append((tail_start, tail_end, f"TAIL (real_len={real_len})"))

    shown = set()
    for start, end, label in ranges:
        new_positions = set(range(start, end)) - shown
        if not new_positions:
            continue
        start = min(new_positions)
        end = max(new_positions) + 1
        shown.update(range(start, end))

        print(f"    [{label}]")
        for j in range(start, end):
            tid = token_ids[j]
            decoded = tok.decode([tid])
            marker = ""
            if tid in image_token_ids:
                marker = "  <-- IMAGE"
            elif tid == action_token_id:
                marker = "  <-- ACTION"
            elif j >= real_len and tid == pad_token_id:
                marker = "  <-- PAD"
            print(f"      {j:5d}: {tid:8d}  {decoded!r}{marker}")
        if end < n:
            print(f"      ...")


def main():
    parser = argparse.ArgumentParser(description="Inspect tokenized sequences for VLA models")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Models to inspect (default: all)")
    parser.add_argument("--samples", nargs="+", type=int, default=[0, 20, 40, 60],
                        help="Dataset sample indices to use (default: 0 20 40 60)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for collation (default: 4)")
    args = parser.parse_args()

    dataset = CALVINDataset(f"{CALVIN_BASE}/training", chunk_size=CHUNK_SIZE)
    print(f"Dataset: {len(dataset)} samples")

    for model_name in args.models:
        try:
            inspect_model(model_name, dataset, args.samples, batch_size=args.batch_size)
        except Exception as e:
            print(f"\n  ERROR inspecting {model_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
