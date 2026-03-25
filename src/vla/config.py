"""Shared constants and model registry for VLA training and inference."""

from dataclasses import dataclass

import torch

ACTION_TOKEN = "<action>"
SYSTEM_PROMPT = (
    "You are a robot manipulation agent. Given an image of the current scene "
    "and a language instruction, predict the next action to execute."
)

ACTION_DIM = 7
CHUNK_SIZE = 10
MAX_LENGTH = 256


@dataclass
class ModelSpec:
    model_id: str
    hidden_dim: int
    model_kwargs: dict
    processor_kwargs: dict
    collate_style: str  # "chat_template" | "paligemma"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "lfm": ModelSpec(
        model_id="LiquidAI/LFM2-VL-3B",
        hidden_dim=2048,
        model_kwargs={"torch_dtype": torch.bfloat16},
        processor_kwargs={"max_image_tokens": 256},
        collate_style="chat_template",
    ),
    "qwen": ModelSpec(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        hidden_dim=2048,
        model_kwargs={"torch_dtype": torch.bfloat16},
        processor_kwargs={"min_pixels": 256 * 28 * 28, "max_pixels": 256 * 28 * 28},
        collate_style="chat_template",
    ),
    "paligemma": ModelSpec(
        model_id="google/paligemma2-3b-mix-224",
        hidden_dim=2048,
        model_kwargs={"torch_dtype": torch.bfloat16},
        processor_kwargs={},
        collate_style="paligemma",
    ),
}
