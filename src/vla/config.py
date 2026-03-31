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
MAX_LENGTH = 512
RGB_PAD = 10
GRIPPER_LOSS_WEIGHT = 0.01


@dataclass
class ModelSpec:
    model_id: str
    hidden_dim: int
    model_kwargs: dict
    processor_kwargs: dict
    collate_style: str  # "chat_template" | "paligemma"
    max_length: int = MAX_LENGTH


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LFM2-VL-3B": ModelSpec(
        model_id="LiquidAI/LFM2-VL-3B",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        collate_style="chat_template",
    ),
    "LFM2-VL-1.6B": ModelSpec(
        model_id="LiquidAI/LFM2-VL-1.6B",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        collate_style="chat_template",
    ),
    "LFM2-VL-450M": ModelSpec(
        model_id="LiquidAI/LFM2-VL-450M",
        hidden_dim=1024,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        collate_style="chat_template",
    ),
    "Qwen2.5-VL-3B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        collate_style="chat_template",
    ),
    "Qwen3-VL-2B-Instruct": ModelSpec(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        collate_style="chat_template",
    ),
    "Qwen3-VL-4B-Instruct": ModelSpec(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        hidden_dim=2560,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        collate_style="chat_template",
    ),
    "paligemma2-3b-mix-224": ModelSpec(
        model_id="google/paligemma2-3b-mix-224",
        hidden_dim=2304,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        collate_style="paligemma",
    ),
    # Base is 2B parameters
    "SmolVLM-Instruct": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-Instruct",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={"size": {"longest_edge": 256}},
        collate_style="chat_template",
    ),
    "SmolVLM-500M-Instruct": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-500M-Instruct",
        hidden_dim=960,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={"size": {"longest_edge": 256}},
        collate_style="chat_template",
    ),
    "SmolVLM-256M-Instruct": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        hidden_dim=576,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={"size": {"longest_edge": 256}},
        collate_style="chat_template",
    ),
}


"""
Image token usage by model (tested in experiments/inspect_tokenized_sequences.py)
LFM2-VL-3B: 64
LFM2-VL-1.6B: 64
Qwen3-VL-2B-Instruct: 64
Qwen3-VL-4B-Instruct: 64
Qwen2.5-VL-3B-Instruct: 49 
paligemma2-3b-mix-224: 256
SmolVLM-Instruct: 81
SmolVLM-500M-Instruct: 64
SmolVLM-256M-Instruct: 64
"""