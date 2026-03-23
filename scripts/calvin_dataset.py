"""CALVIN dataset and collate function for VLM-based behavioral cloning."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ACTION_TOKEN = "<action>"


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
    ):
        self.dataset_dir = Path(dataset_dir)
        self.chunk_size = chunk_size
        self.action_key = action_key
        self.image_key = image_key
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

        # Load action chunk: actions from frame_id to frame_id + chunk_size - 1
        actions = []
        for offset in range(self.chunk_size):
            ep = np.load(self._episode_path(frame_id + offset))
            actions.append(ep[self.action_key])
        action_chunk = torch.tensor(np.stack(actions), dtype=torch.float32)  # (chunk_size, 7)

        return {
            "image": image,
            "instruction": self.texts[ann_idx],
            "action_chunk": action_chunk,
            "frame_id": frame_id,
        }


def make_calvin_collate_fn(processor, system_prompt: str, max_length: int = 256):
    """Returns a collate function that formats CALVIN samples for a VLM.

    Builds chat-template conversations (system + user with image + instruction),
    tokenizes with add_generation_prompt=True, appends the <action> token,
    and stacks ground-truth action chunks.
    """
    action_token_id = processor.tokenizer.convert_tokens_to_ids(ACTION_TOKEN)

    def collate_fn(batch):
        conversations = []
        action_chunks = []

        for sample in batch:
            convo = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": sample["image"]},
                        {"type": "text", "text": sample["instruction"]},
                    ],
                },
            ]
            conversations.append(convo)
            action_chunks.append(sample["action_chunk"])

        vlm_inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Insert <action> token before padding in each sequence
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

        vlm_inputs["gt_actions"] = torch.stack(action_chunks)  # (B, chunk_size, 7)
        return vlm_inputs

    return collate_fn
