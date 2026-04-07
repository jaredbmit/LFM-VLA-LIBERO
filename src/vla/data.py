"""CALVIN dataset and collate function for VLM-based behavioral cloning."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vla.config import ACTION_TOKEN, RGB_PAD, INSTRUCTION_PREPROMPT, NORM_MIN, NORM_MAX


def normalize_action(action: np.ndarray, norm_min: float = NORM_MIN,
                      norm_max: float = NORM_MAX) -> np.ndarray:
    """Clip pose dims to [norm_min, norm_max] and rescale to [-1, 1]. Gripper untouched."""
    out = action.copy()
    out[..., :6] = np.clip(out[..., :6], norm_min, norm_max)
    out[..., :6] = 2 * (out[..., :6] - norm_min) / (norm_max - norm_min) - 1
    return out


def unnormalize_action(action: np.ndarray, norm_min: float = NORM_MIN,
                       norm_max: float = NORM_MAX) -> np.ndarray:
    """Reverse normalize_action for pose dims. Gripper untouched."""
    out = action.copy()
    out[..., :6] = 0.5 * (out[..., :6] + 1) * (norm_max - norm_min) + norm_min
    return out


def random_shift(image: Image.Image, pad: int) -> Image.Image:
    """Random spatial shift with zero-padding (matches VLM4VLA's RandomShiftsAug)."""
    w, h = image.size
    padded = Image.new("RGB", (w + 2 * pad, h + 2 * pad))
    padded.paste(image, (pad, pad))
    dx = random.randint(0, 2 * pad)
    dy = random.randint(0, 2 * pad)
    return padded.crop((dx, dy, dx + w, dy + h))


class CALVINDataset(Dataset):
    """Loads CALVIN language-annotated frames as (image, instruction, action_chunk) samples.

    Each sample is a single frame from a language-annotated segment, paired with
    the annotation text and a chunk of future actions starting at that frame.
    """

    def __init__(
        self,
        dataset_dir: str,
        chunk_size: int = 1,
        action_key: str = "rel_actions",
        image_key: str = "rgb_static",
        rgb_pad: int = 0,
        norm_action: bool = False,
        norm_min: float = NORM_MIN,
        norm_max: float = NORM_MAX,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.chunk_size = chunk_size
        self.action_key = action_key
        self.image_key = image_key
        self.rgb_pad = rgb_pad
        self.norm_action = norm_action
        self.norm_min = norm_min
        self.norm_max = norm_max
        self.sharded = any(self.dataset_dir.glob("ep_*"))

        ann = np.load(
            self.dataset_dir / "lang_annotations" / "auto_lang_ann.npy",
            allow_pickle=True,
        ).item()

        texts = ann["language"]["ann"]
        indx = ann["info"]["indx"]

        # Build flat index, excluding frames too close to the end of a segment
        # to form a full chunk
        self.samples = []
        for ann_idx, (start, end) in enumerate(indx):
            for frame_id in range(start, end + 1 - (chunk_size - 1)):
                self.samples.append((frame_id, ann_idx))

        self.texts = texts

    def _episode_path(self, frame_id: int) -> Path:
        fname = f"episode_{frame_id:07d}.npz"
        if self.sharded:
            return self.dataset_dir / f"ep_{frame_id // 1000:04d}" / fname
        return self.dataset_dir / fname

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_id, ann_idx = self.samples[idx]
        image = Image.fromarray(
            np.load(self._episode_path(frame_id))[self.image_key]
        )
        if self.rgb_pad > 0:
            image = random_shift(image, self.rgb_pad)

        # Load action chunk: actions from frame_id to frame_id + chunk_size - 1
        actions = []
        for offset in range(self.chunk_size):
            ep = np.load(self._episode_path(frame_id + offset))
            actions.append(ep[self.action_key])
        action_chunk = np.stack(actions)  # (chunk_size, 7)
        if self.norm_action:
            action_chunk = normalize_action(action_chunk, self.norm_min, self.norm_max)
        action_chunk = torch.tensor(action_chunk, dtype=torch.float32)

        return {
            "image": image,
            "instruction": self.texts[ann_idx],
            "action_chunk": action_chunk,
            "frame_id": frame_id,
        }


def _insert_token(vlm_inputs: dict, token_id: int, pad_token_id: int) -> dict:
    """Insert a token at the end of each sequence's real content (before padding).

    Expands input_ids, attention_mask, and any other (B, S)-shaped tensors by one
    position per sequence.
    """
    ids = vlm_inputs["input_ids"]
    mask = vlm_inputs["attention_mask"]
    B, S = ids.shape
    new_ids = torch.full((B, S + 1), pad_token_id, dtype=ids.dtype)
    new_mask = torch.zeros((B, S + 1), dtype=mask.dtype)
    seq_keys = [k for k, v in vlm_inputs.items()
                if isinstance(v, torch.Tensor) and v.shape == (B, S)
                and k not in ("input_ids", "attention_mask")]
    new_seq = {k: torch.zeros((B, S + 1), dtype=vlm_inputs[k].dtype) for k in seq_keys}
    for i in range(B):
        real_len = mask[i].sum().item()
        new_ids[i, :real_len] = ids[i, :real_len]
        new_ids[i, real_len] = token_id
        new_ids[i, real_len + 1:] = ids[i, real_len:]
        new_mask[i, :real_len + 1] = 1
        for k in seq_keys:
            new_seq[k][i, :real_len] = vlm_inputs[k][i, :real_len]
            new_seq[k][i, real_len + 1:] = vlm_inputs[k][i, real_len:]
    vlm_inputs["input_ids"] = new_ids
    vlm_inputs["attention_mask"] = new_mask
    for k in seq_keys:
        vlm_inputs[k] = new_seq[k]
    return vlm_inputs


def make_calvin_collate_fn(processor, system_prompt: str, max_length: int = 512,
                           collate_style: str = "chat_template"):
    """Returns a collate function that formats CALVIN samples for a VLM.

    Both styles tokenize the image + instruction, then insert the <action> token
    post-tokenization at the end of real content (before padding).

    collate_style="chat_template": builds system+user conversations, tokenizes via
        apply_chat_template (LFM, Qwen).
    collate_style="paligemma": uses the PaliGemma processor directly.
    """
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    action_token_id = tok.convert_tokens_to_ids(ACTION_TOKEN)

    def _tokenize_chat_template(batch):
        conversations = []
        for sample in batch:
            conversations.append([
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": INSTRUCTION_PREPROMPT + sample["instruction"]},
                ]},
            ])
        return processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    def _tokenize_paligemma(batch):
        texts = [f"<image>{s['instruction']}\n" for s in batch]
        return processor(
            text=texts,
            images=[s["image"] for s in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

    tokenize = _tokenize_chat_template if collate_style == "chat_template" else _tokenize_paligemma

    def collate(batch):
        vlm_inputs = tokenize(batch)
        _insert_token(vlm_inputs, action_token_id, tok.pad_token_id)
        vlm_inputs["gt_actions"] = torch.stack([s["action_chunk"] for s in batch])
        return vlm_inputs

    return collate
