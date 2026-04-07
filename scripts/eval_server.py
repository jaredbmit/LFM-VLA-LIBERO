"""VLA inference server: loads a trained checkpoint and serves actions over TCP.

Run in the LFM-VLA venv. Communicates with eval_client.py via JSON-over-TCP.

Supports batched inference: N worker clients connect simultaneously, the server
collects their requests into a single batch, runs one forward pass, and returns
results to each client. Set --num_workers to match the client's --num_workers.

Usage:
    uv run python scripts/eval_server.py \
        --checkpoint runs/<timestamp>/checkpoints/best \
        --num_workers 4
"""

import argparse
import base64
import io
import json
import queue
import socket
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vla import VLA, ACTION_TOKEN, CHUNK_SIZE, MODEL_REGISTRY, SYSTEM_PROMPT
from vla.config import ACTION_DIM
from vla.data import make_calvin_collate_fn, unnormalize_action


def load_checkpoint(ckpt_dir: Path, device: str, spec):
    print(f"Loading processor from {ckpt_dir / 'vlm'}...")
    processor = AutoProcessor.from_pretrained(ckpt_dir / "vlm")
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tok.padding_side = "right"

    print(f"Loading VLM from {ckpt_dir / 'vlm'}...")
    vlm = AutoModelForImageTextToText.from_pretrained(
        ckpt_dir / "vlm", device_map=device, torch_dtype=torch.bfloat16,
    )

    action_token_id = tok.convert_tokens_to_ids(ACTION_TOKEN)
    vla = VLA(vlm, action_token_id=action_token_id, hidden_dim=spec.hidden_dim).to(device)

    ckpt = torch.load(ckpt_dir / "action_head.pt", map_location=device, weights_only=True)
    vla.proj.load_state_dict(ckpt["proj"])
    vla.pose_head.load_state_dict(ckpt["pose_head"])
    vla.gripper_head.load_state_dict(ckpt["gripper_head"])
    vla.eval()

    # Load hparams to recover normalization settings
    hparams_path = ckpt_dir.parent.parent / "hparams.json"
    hparams = {}
    if hparams_path.exists():
        with open(hparams_path) as f:
            hparams = json.load(f)

    print(f"Loaded checkpoint from step {ckpt['step']}, val_loss={ckpt['val_loss']:.6f}")
    if hparams.get("norm_action"):
        print(f"  norm_action=True, norm_min={hparams['norm_min']}, norm_max={hparams['norm_max']}")
    return vla, processor, hparams


def postprocess(pred_chunk: torch.Tensor,
                norm_action: bool = False,
                norm_min: float = -0.65,
                norm_max: float = 0.65) -> list[list[float]]:
    """Convert raw model output to CALVIN-compatible actions.

    pred_chunk: (chunk_size, 7) — dims 0-5 are pose deltas, dim 6 is gripper logit.
    If norm_action is True, pose dims are unnormalized from [-1,1] back to
    [norm_min, norm_max] before being sent to the environment.
    """
    actions = []
    for t in range(pred_chunk.shape[0]):
        a = pred_chunk[t].cpu().numpy().copy()
        if norm_action:
            a = unnormalize_action(a, norm_min, norm_max)
        a[:6] = np.clip(a[:6], -1.0, 1.0)
        gripper_prob = 1.0 / (1.0 + np.exp(-float(a[6])))
        a[6] = 1.0 if gripper_prob > 0.5 else -1.0
        actions.append(a.tolist())
    return actions


class BatchInferenceEngine:
    """Collects requests from concurrent worker threads and runs batched inference.

    Each worker calls submit() which blocks until the batch is processed.
    The inference loop fires when max_batch_size requests are queued or
    batch_timeout_s seconds have elapsed since the first request arrived.
    """

    def __init__(self, vla, collate_fn, device: str, max_batch_size: int,
                 batch_timeout_s: float = 0.05,
                 norm_action: bool = False,
                 norm_min: float = -0.65, norm_max: float = 0.65):
        self.vla = vla
        self.collate_fn = collate_fn
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_timeout_s = batch_timeout_s
        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self._queue: queue.Queue = queue.Queue()

    def submit(self, image: Image.Image, instruction: str) -> list[list[float]]:
        """Submit a single request; block until the batch result is ready."""
        event = threading.Event()
        result: list = [None]
        self._queue.put((image, instruction, event, result))
        event.wait()
        return result[0]

    def run(self):
        """Inference loop — run in a dedicated thread."""
        while True:
            # Block until at least one request arrives
            try:
                first = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            pending = [first]
            deadline = time.monotonic() + self.batch_timeout_s

            # Collect more requests up to max_batch_size or timeout
            while len(pending) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    pending.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                # Build batch
                dummy_chunk = torch.zeros(CHUNK_SIZE, ACTION_DIM)
                samples = [
                    {"image": img, "instruction": instr, "action_chunk": dummy_chunk}
                    for img, instr, _, _ in pending
                ]
                vlm_inputs = self.collate_fn(samples)
                vlm_inputs.pop("gt_actions")
                vlm_inputs = {k: v.to(self.device) for k, v in vlm_inputs.items()}

                with torch.inference_mode():
                    preds = self.vla(**vlm_inputs)  # (B, chunk_size, 7)

                # Return results to each waiting thread
                for i, (_, _, event, result) in enumerate(pending):
                    result[0] = postprocess(preds[i], self.norm_action,
                                                        self.norm_min, self.norm_max)
                    event.set()

            except Exception as e:
                import traceback
                print(f"Inference error (batch={len(pending)}): {e}", flush=True)
                traceback.print_exc()
                # Unblock waiting workers so they don't hang forever
                for _, _, event, result in pending:
                    result[0] = None
                    event.set()


def recv_line(conn: socket.socket, buf: bytearray) -> str | None:
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return None
        buf.extend(chunk)
    idx = buf.index(b"\n")
    line = buf[:idx].decode("utf-8")
    del buf[:idx + 1]
    return line


def handle_connection(conn: socket.socket, engine: BatchInferenceEngine,
                      shutdown_event: threading.Event):
    buf = bytearray()
    try:
        while not shutdown_event.is_set():
            line = recv_line(conn, buf)
            if line is None:
                break
            request = json.loads(line)
            if request.get("shutdown"):
                shutdown_event.set()
                conn.sendall(json.dumps({"status": "shutdown"}).encode() + b"\n")
                break
            image = Image.open(io.BytesIO(base64.b64decode(request["image"]))).convert("RGB")
            actions = engine.submit(image, request["instruction"])
            conn.sendall(json.dumps({"actions": actions}).encode() + b"\n")
    except Exception as e:
        print(f"Connection error: {e}")
    finally:
        conn.close()
        print("Client disconnected.")


def serve(vla, collate_fn, device: str, port: int, num_workers: int,
          norm_action: bool = False, norm_min: float = -0.65, norm_max: float = 0.65):
    engine = BatchInferenceEngine(vla, collate_fn, device, max_batch_size=num_workers,
                                  norm_action=norm_action, norm_min=norm_min,
                                  norm_max=norm_max)
    inference_thread = threading.Thread(target=engine.run, daemon=True)
    inference_thread.start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("localhost", port))
    server.listen(num_workers)
    server.settimeout(1.0)
    print(f"VLA inference server listening on localhost:{port} (max_batch={num_workers})")

    shutdown_event = threading.Event()
    threads = []
    try:
        while not shutdown_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            print(f"Client connected from {addr}")
            t = threading.Thread(target=handle_connection,
                                 args=(conn, engine, shutdown_event), daemon=True)
            t.start()
            threads.append(t)
    finally:
        server.close()
        for t in threads:
            t.join(timeout=2.0)


def main():
    parser = argparse.ArgumentParser(description="VLA inference server")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    parser.add_argument("--model", default="lfm", choices=list(MODEL_REGISTRY),
                        help="VLM backbone used during training (default: lfm2)")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel eval worker clients (sets max batch size)")
    args = parser.parse_args()

    spec = MODEL_REGISTRY[args.model]
    vla, processor, hparams = load_checkpoint(Path(args.checkpoint), args.device, spec)
    collate_fn = make_calvin_collate_fn(processor, SYSTEM_PROMPT,
                                        max_length=spec.max_length,
                                        collate_style=spec.collate_style)
    serve(vla, collate_fn, args.device, args.port, args.num_workers,
          norm_action=hparams.get("norm_action", False),
          norm_min=hparams.get("norm_min", -0.65),
          norm_max=hparams.get("norm_max", 0.65))


if __name__ == "__main__":
    main()
