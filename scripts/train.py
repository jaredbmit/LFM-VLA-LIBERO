"""Train a VLA policy on CALVIN with behavioral cloning."""

import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor

from vla import VLA, FlowMatchingVLA, ACTION_TOKEN, CHUNK_SIZE, MODEL_REGISTRY, SYSTEM_PROMPT
from vla.model import masked_action_mse
from vla.config import ACTION_DIM, RGB_PAD, NORM_ACTION, NORM_MIN, NORM_MAX
from vla.data import CALVINDataset, make_calvin_collate_fn

# CALVIN_BASE = "/home/jared/drl/calvin/dataset/calvin_debug_dataset"
# RUN_DIR = "/home/jared/lfm-vla/runs"

CALVIN_BASE = "/home/schmidt/ssci-jaredb/scratch_ssci-rus/jaredb/datasets/CALVIN/task_ABC_D_annotated"
RUN_DIR = "/home/schmidt/ssci-jaredb/scratch_ssci-rus/jaredb/runs/v6"


# HPPs
NUM_STEPS = 30000
LOG_EVERY = 100
EVAL_EVERY = 3000
SAVE_EVERY = 1e8  # unused
MAX_VAL_BATCHES = 500
LR = 2e-5
WEIGHT_DECAY = 1e-4
WARMUP_STEPS = 1000
LR_SCHEDULE = "constant"  # "constant" | "cosine"
GRAD_CLIP = 1.0


def save_checkpoint(run_dir: Path, tag: str, vla, processor, step, val_loss):
    ckpt_dir = run_dir / "checkpoints" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"step": step, "val_loss": val_loss, **vla.head_state_dict()},
               ckpt_dir / "action_head.pt")
    vla.vlm.save_pretrained(ckpt_dir / "vlm")
    processor.save_pretrained(ckpt_dir / "vlm")

    print(f"  Saved checkpoint: {ckpt_dir}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    run_dir = Path(RUN_DIR) / f"{timestamp}_{uid}"
    run_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train a VLA policy on CALVIN")
    parser.add_argument("--model", default="LFM2-VL-450M", choices=list(MODEL_REGISTRY),
                        help="VLM backbone to use (default: LFM2-VL-450M)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lr_schedule", choices=["constant", "cosine"], default=LR_SCHEDULE)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--norm_action", action=argparse.BooleanOptionalAction, default=NORM_ACTION)
    parser.add_argument("--norm_min", type=float, default=NORM_MIN)
    parser.add_argument("--norm_max", type=float, default=NORM_MAX)
    parser.add_argument("--head", choices=["mlp", "flow"], default="mlp",
                        help="Action head type: mlp (default) or flow (flow matching transformer)")
    parser.add_argument("--flow_steps", type=int, default=10,
                        help="Denoising steps for inference latency benchmark (flow head only)")
    parser.add_argument("--d_model", type=int, default=256,
                        help="Flow head transformer width (flow head only)")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Flow head attention heads (flow head only)")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Flow head transformer depth (flow head only)")
    args = parser.parse_args()
    spec = MODEL_REGISTRY[args.model]
    batch_size = args.batch_size if args.batch_size is not None else spec.default_batch_size
    grad_steps = args.grad_steps if args.grad_steps is not None else spec.default_grad_steps

    hparams = {
        "model": args.model, "model_id": spec.model_id,
        "calvin_base": CALVIN_BASE,
        "batch_size": batch_size, "grad_steps": grad_steps, "num_steps": NUM_STEPS,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule, "warmup_steps": args.warmup_steps,
        "grad_clip": GRAD_CLIP,
        "action_dim": ACTION_DIM, "chunk_size": CHUNK_SIZE, "max_length": spec.max_length,
        "norm_action": args.norm_action, "norm_min": args.norm_min, "norm_max": args.norm_max,
        "head": args.head,
        **({"d_model": args.d_model, "n_heads": args.n_heads, "n_layers": args.n_layers,
            "flow_steps": args.flow_steps} if args.head == "flow" else {}),
    }
    with open(run_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    log_path = run_dir / "metrics.csv"
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["step", "train_loss", "val_loss", "elapsed_sec", "gpu_mem_mib"])

    print(f"Run directory: {run_dir}")
    print(f"Model: {spec.model_id}")
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(spec.model_id, **spec.processor_kwargs)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tok.padding_side = "right"
    tok.add_special_tokens({"additional_special_tokens": [ACTION_TOKEN]})

    print("Loading VLM...")
    vlm = AutoModelForImageTextToText.from_pretrained(
        spec.model_id, **spec.model_kwargs,
    ).to("cuda")
    vlm.resize_token_embeddings(len(tok))

    device = next(vlm.parameters()).device
    action_token_id = tok.convert_tokens_to_ids(ACTION_TOKEN)
    if args.head == "flow":
        vla = FlowMatchingVLA(
            vlm, hidden_dim=spec.hidden_dim, chunk_size=CHUNK_SIZE,
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            n_steps_inference=args.flow_steps,
        ).to(device)
        print(f"Flow head fusion layer: {vla.fusion_layer_idx} (~70% depth)")
    else:
        vla = VLA(vlm, action_token_id=action_token_id, hidden_dim=spec.hidden_dim,
                  chunk_size=CHUNK_SIZE).to(device)

    trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vla.parameters())
    print(f"Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")

    train_ds = CALVINDataset(f"{CALVIN_BASE}/training", chunk_size=CHUNK_SIZE,
                             rgb_pad=RGB_PAD,
                             norm_action=args.norm_action,
                             norm_min=args.norm_min, norm_max=args.norm_max)
    val_ds = CALVINDataset(f"{CALVIN_BASE}/validation", chunk_size=CHUNK_SIZE,
                           norm_action=args.norm_action,
                           norm_min=args.norm_min, norm_max=args.norm_max)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    collate_fn = make_calvin_collate_fn(processor, SYSTEM_PROMPT, max_length=spec.max_length,
                                        collate_style=spec.collate_style)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=NUM_STEPS,
            pct_start=args.warmup_steps / NUM_STEPS, anneal_strategy="cos",
        )
    else:  # "constant"
        from transformers.optimization import get_constant_schedule_with_warmup
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=args.warmup_steps)
    print(f"\nTraining: {len(train_ds)} samples, {len(val_ds)} val, "
          f"{NUM_STEPS} steps, grad_steps={grad_steps}, chunk_size={CHUNK_SIZE}\n")

    vla.train()
    optimizer.zero_grad()
    train_iter = iter(train_loader)
    running_loss = 0.0  # sum of raw losses over micro-steps, reset every LOG_EVERY optimizer steps
    update_step = 1
    best_val_loss = float("inf")
    start_time = time.time()

    for micro_step in range(1, NUM_STEPS * grad_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        gt_actions = batch.pop("gt_actions").to(device)
        act_mask = batch.pop("action_mask").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        loss = vla.compute_loss(batch, gt_actions, act_mask)
        (loss / grad_steps).backward()
        running_loss += loss.item()

        if micro_step % grad_steps != 0:
            continue

        torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if update_step % LOG_EVERY == 0:
            avg_loss = running_loss / (LOG_EVERY * grad_steps)
            elapsed = time.time() - start_time
            gpu_mib = torch.cuda.max_memory_allocated(device) // (1024 * 1024)
            torch.cuda.reset_peak_memory_stats(device)
            print(f"  step {update_step:5d}  train_loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}  gpu={gpu_mib}MiB  [{elapsed:.0f}s]")
            csv_writer.writerow([update_step, f"{avg_loss:.6f}", "", f"{elapsed:.1f}", gpu_mib])
            log_file.flush()
            running_loss = 0.0

        if update_step % EVAL_EVERY == 0:
            vla.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    gt = val_batch.pop("gt_actions").to(device)
                    val_mask = val_batch.pop("action_mask").to(device)
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    pred = vla.predict_actions(**val_batch)
                    val_loss_sum += masked_action_mse(pred, gt, val_mask).item()
                    val_steps += 1
                    if val_steps >= MAX_VAL_BATCHES:
                        break
            val_loss = val_loss_sum / val_steps
            elapsed = time.time() - start_time
            print(f"           val_loss={val_loss:.6f}")
            csv_writer.writerow([update_step, "", f"{val_loss:.6f}", f"{elapsed:.1f}"])
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(run_dir, "best", vla, processor, update_step, val_loss)

            vla.train()

        if update_step % SAVE_EVERY == 0:
            save_checkpoint(run_dir, f"step_{update_step}", vla, processor, update_step, best_val_loss)
            
        update_step += 1

    save_checkpoint(run_dir, "final", vla, processor, NUM_STEPS, best_val_loss)

    elapsed = time.time() - start_time
    print(f"\nTotal training time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Run saved to: {run_dir}")

    # Inference latency benchmark (batch size 1)
    print("\nMeasuring inference latency (batch_size=1)...")
    vla.eval()
    single_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    single_batch = next(iter(single_loader))
    single_batch.pop("gt_actions")
    single_batch.pop("action_mask")
    single_batch = {k: v.to(device) for k, v in single_batch.items()}

    WARMUP = 10
    MEASURE = 50
    with torch.no_grad():
        for _ in range(WARMUP):
            vla(**single_batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(MEASURE):
            vla(**single_batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    ms_per_step = (t1 - t0) / MEASURE * 1000
    print(f"Inference latency: {ms_per_step:.1f} ms/sample ({1000/ms_per_step:.1f} Hz)")

    with open(run_dir / "inference_latency.json", "w") as f:
        json.dump({"ms_per_sample": round(ms_per_step, 3), "hz": round(1000 / ms_per_step, 2),
                   "warmup_steps": WARMUP, "measure_steps": MEASURE}, f, indent=2)

    log_file.close()


if __name__ == "__main__":
    main()
