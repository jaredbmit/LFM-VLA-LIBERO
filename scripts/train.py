"""Train a VLA policy on CALVIN with behavioral cloning."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from vla import VLA, ACTION_TOKEN, CHUNK_SIZE, MAX_LENGTH, SYSTEM_PROMPT
from vla.config import ACTION_DIM, HIDDEN_DIM
from vla.data import CALVINDataset, make_calvin_collate_fn

CALVIN_BASE = "/home/jared/drl/calvin/dataset/calvin_debug_dataset"
MODEL_ID = "LiquidAI/LFM2-VL-3B"

BATCH_SIZE = 4
NUM_STEPS = 10000
LOG_EVERY = 100
EVAL_EVERY = 500
SAVE_EVERY = 1000
LR = 1e-5
RUN_DIR = "runs"


def save_checkpoint(run_dir: Path, tag: str, vla, optimizer, processor, step, val_loss):
    ckpt_dir = run_dir / "checkpoints" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "step": step,
        "val_loss": val_loss,
        "action_head": vla.action_head.state_dict(),
    }, ckpt_dir / "action_head.pt")

    vla.vlm.save_pretrained(ckpt_dir / "vlm")
    processor.save_pretrained(ckpt_dir / "vlm")

    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")

    print(f"  Saved checkpoint: {ckpt_dir}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RUN_DIR) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    hparams = {
        "calvin_base": CALVIN_BASE, "model_id": MODEL_ID,
        "batch_size": BATCH_SIZE, "num_steps": NUM_STEPS, "lr": LR,
        "hidden_dim": HIDDEN_DIM, "action_dim": ACTION_DIM,
        "chunk_size": CHUNK_SIZE, "max_length": MAX_LENGTH,
    }
    with open(run_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    log_path = run_dir / "metrics.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["step", "train_loss", "val_loss", "elapsed_sec"])

    print(f"Run directory: {run_dir}")
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_image_tokens=256)
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [ACTION_TOKEN]})

    print("Loading VLM...")
    vlm = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="auto", dtype=torch.bfloat16,
    )
    vlm.resize_token_embeddings(len(processor.tokenizer))

    device = next(vlm.parameters()).device
    action_token_id = processor.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)
    vla = VLA(vlm, action_token_id=action_token_id, chunk_size=CHUNK_SIZE).to(device)

    trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vla.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")

    train_ds = CALVINDataset(f"{CALVIN_BASE}/training", chunk_size=CHUNK_SIZE)
    val_ds = CALVINDataset(f"{CALVIN_BASE}/validation", chunk_size=CHUNK_SIZE)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    collate_fn = make_calvin_collate_fn(processor, SYSTEM_PROMPT, max_length=MAX_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(vla.parameters(), lr=LR)
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    def loss_fn(pred, gt):
        pose_loss = mse_fn(pred[:, :, :6], gt[:, :, :6])
        gripper_target = (gt[:, :, 6] + 1) / 2
        gripper_loss = bce_fn(pred[:, :, 6], gripper_target)
        return pose_loss + gripper_loss

    print(f"\nTraining: {len(train_ds)} samples, {len(val_ds)} val, "
          f"{NUM_STEPS} steps, chunk_size={CHUNK_SIZE}\n")

    vla.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()

    for step in range(1, NUM_STEPS + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        gt_actions = batch.pop("gt_actions").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        pred_actions = vla(**batch)
        loss = loss_fn(pred_actions, gt_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % LOG_EVERY == 0:
            avg_loss = running_loss / LOG_EVERY
            elapsed = time.time() - start_time
            print(f"  step {step:5d}  train_loss={avg_loss:.6f}  [{elapsed:.0f}s]")
            csv_writer.writerow([step, f"{avg_loss:.6f}", "", f"{elapsed:.1f}"])
            log_file.flush()
            running_loss = 0.0

        if step % EVAL_EVERY == 0:
            vla.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    gt = val_batch.pop("gt_actions").to(device)
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    val_loss_sum += loss_fn(vla(**val_batch), gt).item()
                    val_steps += 1
            val_loss = val_loss_sum / val_steps
            elapsed = time.time() - start_time
            print(f"           val_loss={val_loss:.6f}")
            csv_writer.writerow([step, "", f"{val_loss:.6f}", f"{elapsed:.1f}"])
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(run_dir, "best", vla, optimizer, processor, step, val_loss)

            vla.train()

        if step % SAVE_EVERY == 0:
            save_checkpoint(run_dir, f"step_{step}", vla, optimizer, processor, step, best_val_loss)

    save_checkpoint(run_dir, "final", vla, optimizer, processor, NUM_STEPS, best_val_loss)

    log_file.close()

    # Final check
    vla.eval()
    batch = next(iter(val_loader))
    gt_actions = batch.pop("gt_actions").to(device)
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        pred_actions = vla(**batch)

    print(f"\nSample predictions (first action in chunk) vs ground truth:")
    for i in range(min(4, pred_actions.shape[0])):
        pred = pred_actions[i, 0].cpu().tolist()
        gt = gt_actions[i, 0].cpu().tolist()
        print(f"  [{i}] pred: [{', '.join(f'{x:+.3f}' for x in pred)}]")
        print(f"      gt:   [{', '.join(f'{x:+.3f}' for x in gt)}]")

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Run saved to: {run_dir}")


if __name__ == "__main__":
    main()
