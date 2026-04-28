"""Shared constants and model registry for VLA training and inference."""

from dataclasses import dataclass

import torch

ACTION_TOKEN = "<action>"
SYSTEM_PROMPT = "You are a helpful assistant."
INSTRUCTION_PREPROMPT = "What action should the robotic arm take to "

ACTION_DIM = 7
CHUNK_SIZE = 10
MAX_LENGTH = 512
RGB_PAD = 10
IMAGE_SIZE = 224
GRIPPER_LOSS_WEIGHT = 0.01

# Action normalization: clip pose dims to [NORM_MIN, NORM_MAX] then rescale to [-1, 1].
# Gripper dim is left untouched.  Set NORM_ACTION=False to disable.
NORM_ACTION = True
NORM_MIN = -0.65
NORM_MAX = 0.65


# Freeze-group registry: dotted module paths under `vla.vlm`
LFM_GROUPS = {
    "vision": ("model.vision_tower",),
    "connector": ("model.multi_modal_projector",),
    "language": ("model.language_model", "lm_head"),
}
QWEN_VL_GROUPS = {
    "vision": ("model.visual",),
    "connector": (),
    "language": ("model.language_model", "lm_head"),
}
SMOLVLM_GROUPS = {
    "vision": ("model.vision_model",),
    "connector": ("model.connector",),
    "language": ("model.text_model", "lm_head"),
}


@dataclass
class ModelSpec:
    model_id: str
    hidden_dim: int
    model_kwargs: dict
    processor_kwargs: dict
    freeze_groups: dict
    max_length: int = MAX_LENGTH
    default_batch_size: int = 128  # models >= 3B use 64
    default_grad_steps: int = 1   # models >= 3B use 2


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "LFM2-VL-3B": ModelSpec(
        model_id="LiquidAI/LFM2-VL-3B",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=LFM_GROUPS,
        default_batch_size=64,
        default_grad_steps=2,
    ),
    "LFM2-VL-1.6B": ModelSpec(
        model_id="LiquidAI/LFM2-VL-1.6B",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=LFM_GROUPS,
    ),
    "LFM2-VL-450M": ModelSpec(
        model_id="LiquidAI/LFM2-VL-450M",
        hidden_dim=1024,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=LFM_GROUPS,
    ),
    "LFM2.5-VL-1.6B": ModelSpec(
        model_id="/home/schmidt/ssci-jaredb/scratch_ssci-rus/jaredb/models/LFM2.5-VL-1.6B",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=LFM_GROUPS,
    ),
    "LFM2.5-VL-450M": ModelSpec(
        model_id="/home/schmidt/ssci-jaredb/scratch_ssci-rus/jaredb/models/LFM2.5-VL-450M",
        hidden_dim=1024,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=LFM_GROUPS,
    ),
    "LFM2.5-VL-450M-Grounding-v1": ModelSpec(
        model_id="/home/schmidt/ssci-jaredb/scratch_ssci-rus/jaredb/models/LFM2.5-VL-450M-Grounding-v1",
        hidden_dim=1024,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=LFM_GROUPS,
    ),
    "LFM2.5-VL-450M-Grounding-v2": ModelSpec(
        model_id="/home/schmidt/ssci-jaredb/scratch_ssci-rus/jaredb/models/LFM2.5-VL-450M-Grounding-v2",
        hidden_dim=1024,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=LFM_GROUPS,
    ),
    "LFM2.5-VL-450M-Grounding-v3": ModelSpec(
        model_id="/home/schmidt/ssci-jaredb/scratch_ssci-rus/jaredb/models/LFM2.5-VL-450M-Grounding-v3",
        hidden_dim=1024,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=LFM_GROUPS,
    ),
    "Qwen2.5-VL-3B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=QWEN_VL_GROUPS,
        default_batch_size=64,
        default_grad_steps=2,
    ),
    "Qwen3-VL-2B-Instruct": ModelSpec(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=QWEN_VL_GROUPS,
    ),
    "Qwen3-VL-4B-Instruct": ModelSpec(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        hidden_dim=2560,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        freeze_groups=QWEN_VL_GROUPS,
        default_batch_size=64,
        default_grad_steps=2,
    ),
    # Base is 2B parameters
    "SmolVLM-Instruct": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-Instruct",
        hidden_dim=2048,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={"size": {"longest_edge": 256}},
        freeze_groups=SMOLVLM_GROUPS,
    ),
    "SmolVLM-500M-Instruct": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-500M-Instruct",
        hidden_dim=960,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={"size": {"longest_edge": 256}},
        freeze_groups=SMOLVLM_GROUPS,
    ),
    "SmolVLM-256M-Instruct": ModelSpec(
        model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        hidden_dim=576,
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={"size": {"longest_edge": 256}},
        freeze_groups=SMOLVLM_GROUPS,
    ),
}
