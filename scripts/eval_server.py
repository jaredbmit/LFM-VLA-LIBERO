"""VLA inference server: loads a trained checkpoint and serves actions over TCP.

Run in the LFM-VLA venv. Communicates with eval_client.py via JSON-over-TCP.

Usage:
    uv run python scripts/eval_server.py \
        --checkpoint runs/<timestamp>/checkpoints/best
"""

import argparse
import base64
import io
import json
import socket
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from vla import VLA, ACTION_TOKEN, CHUNK_SIZE, MAX_LENGTH, SYSTEM_PROMPT
from vla.config import ACTION_DIM, HIDDEN_DIM


def load_checkpoint(ckpt_dir: Path, device: str):
    print(f"Loading processor from {ckpt_dir / 'vlm'}...")
    processor = AutoProcessor.from_pretrained(ckpt_dir / "vlm", max_image_tokens=256)
    processor.tokenizer.padding_side = "right"

    print(f"Loading VLM from {ckpt_dir / 'vlm'}...")
    vlm = AutoModelForImageTextToText.from_pretrained(
        ckpt_dir / "vlm", device_map=device, dtype=torch.bfloat16,
    )

    action_token_id = processor.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)
    vla = VLA(vlm, action_token_id=action_token_id).to(device)

    ckpt = torch.load(ckpt_dir / "action_head.pt", map_location=device, weights_only=True)
    vla.action_head.load_state_dict(ckpt["action_head"])
    vla.eval()

    print(f"Loaded checkpoint from step {ckpt['step']}, val_loss={ckpt['val_loss']:.6f}")
    return vla, processor


def preprocess(image: Image.Image, instruction: str, processor, device: str):
    """Build VLM inputs from a single image + instruction, matching training collate."""
    action_token_id = processor.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        },
    ]

    vlm_inputs = processor.apply_chat_template(
        [conversation],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Append <action> token (same logic as collate in vla/data.py)
    ids = vlm_inputs["input_ids"]
    mask = vlm_inputs["attention_mask"]
    B, S = ids.shape
    new_ids = torch.full((B, S + 1), processor.tokenizer.pad_token_id, dtype=ids.dtype)
    new_mask = torch.zeros((B, S + 1), dtype=mask.dtype)
    for i in range(B):
        real_len = mask[i].sum().item()
        new_ids[i, :real_len] = ids[i, :real_len]
        new_ids[i, real_len] = action_token_id
        new_ids[i, real_len + 1:] = ids[i, real_len:]
        new_mask[i, :real_len + 1] = 1
    vlm_inputs["input_ids"] = new_ids
    vlm_inputs["attention_mask"] = new_mask

    return {k: v.to(device) for k, v in vlm_inputs.items()}


def postprocess(pred_chunk: torch.Tensor) -> list[list[float]]:
    """Convert raw model output to CALVIN-compatible actions.

    pred_chunk: (chunk_size, 7) — dims 0-5 are pose deltas, dim 6 is gripper logit.
    """
    actions = []
    for t in range(pred_chunk.shape[0]):
        a = pred_chunk[t].cpu().numpy().copy()
        a[:6] = np.clip(a[:6], -1.0, 1.0)
        gripper_prob = 1.0 / (1.0 + np.exp(-float(a[6])))
        a[6] = 1.0 if gripper_prob > 0.5 else -1.0
        actions.append(a.tolist())
    return actions


def handle_request(data: dict, vla, processor, device: str) -> dict:
    img_bytes = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    instruction = data["instruction"]

    vlm_inputs = preprocess(image, instruction, processor, device)

    with torch.no_grad():
        pred = vla(**vlm_inputs)  # (1, chunk_size, 7)

    actions = postprocess(pred[0])
    return {"actions": actions}


def recv_line(conn: socket.socket, buf: bytearray) -> str | None:
    """Read one newline-delimited JSON message from the socket."""
    while b"\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return None
        buf.extend(chunk)
    idx = buf.index(b"\n")
    line = buf[:idx].decode("utf-8")
    del buf[:idx + 1]
    return line


def serve(vla, processor, device: str, port: int):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("localhost", port))
    server.listen(1)
    print(f"VLA inference server listening on localhost:{port}")

    while True:
        conn, addr = server.accept()
        print(f"Client connected from {addr}")
        buf = bytearray()
        try:
            while True:
                line = recv_line(conn, buf)
                if line is None:
                    break
                request = json.loads(line)
                if request.get("shutdown"):
                    print("Shutdown requested.")
                    conn.sendall(json.dumps({"status": "shutdown"}).encode() + b"\n")
                    conn.close()
                    server.close()
                    return
                response = handle_request(request, vla, processor, device)
                conn.sendall(json.dumps(response).encode() + b"\n")
        except Exception as e:
            print(f"Error handling request: {e}")
        finally:
            conn.close()
            print("Client disconnected.")


def main():
    parser = argparse.ArgumentParser(description="VLA inference server")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    vla, processor = load_checkpoint(Path(args.checkpoint), args.device)
    serve(vla, processor, args.device, args.port)


if __name__ == "__main__":
    main()
