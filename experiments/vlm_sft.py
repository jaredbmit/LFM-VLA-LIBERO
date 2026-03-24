import os
import subprocess
import threading
import time

os.environ["WANDB_DISABLED"] = "true"

from dataclasses import dataclass
from typing import Callable

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainerCallback
from trl import SFTConfig, SFTTrainer


@dataclass
class ModelSpec:
    model_id: str
    model_kwargs: dict
    processor_kwargs: dict
    lora_targets: str | tuple
    processing_class: str   # "tokenizer" or "processor"
    collate_style: str      # "chat_template" or "paligemma"


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "lfm2": ModelSpec(
        model_id="LiquidAI/LFM2-VL-3B",
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={"max_image_tokens": 256},
        lora_targets="all-linear",
        processing_class="tokenizer",
        collate_style="chat_template",
    ),
    "qwen": ModelSpec(
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={"min_pixels": 256 * 28 * 28, "max_pixels": 256 * 28 * 28},
        lora_targets="all-linear",
        processing_class="processor",
        collate_style="chat_template",
    ),
    "paligemma": ModelSpec(
        model_id="google/paligemma2-3b-mix-224",
        model_kwargs={"dtype": torch.bfloat16},
        processor_kwargs={},
        lora_targets="all-linear",
        processing_class="processor",
        collate_style="paligemma",
    ),
}


MED_VQA_SYSTEM = (
    "You are a medical Vision Language Model specialized in analyzing medical images and providing clinical insights. "
    "Provide concise, clinically relevant answers based on the image and question."
)

VQA_SYSTEM = "Answer the question about the image concisely."


def format_med_vqa_sample(sample):
    return [
        {"role": "system", "content": [{"type": "text", "text": MED_VQA_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["gt_answer"]}]},
    ]


def format_vqav2_sample(sample):
    return [
        {"role": "system", "content": [{"type": "text", "text": VQA_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": sample["question"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["multiple_choice_answer"]}]},
    ]


@dataclass
class DatasetSpec:
    dataset_id: str
    source_split: str
    format_fn: Callable


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "med-vqa": DatasetSpec(
        dataset_id="simwit/omni-med-vqa-mini",
        source_split="test",
        format_fn=format_med_vqa_sample,
    ),
    "vqav2": DatasetSpec(
        dataset_id="Multimodal-Fatima/VQAv2_train",
        source_split="train",
        format_fn=format_vqav2_sample,
    ),
}


@dataclass
class TrainConfig:
    model_name: str = "lfm2"
    dataset_name: str = "vqav2"
    output_dir: str | None = None  # auto-derived as "{model_name}-{dataset_name}" if None
    eval_size: float = 0.2
    seed: int = 42
    max_train_samples: int | None = 2000   # None = use full split
    max_eval_samples: int | None = 500     # None = use full split
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    min_lr_ratio: float = 1e-7 / 3e-5  # cosine decay floor (LFM2 paper: 1e-7)
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    max_length: int = 512
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def __post_init__(self):
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_name {self.model_name!r}. Choose from: {list(MODEL_REGISTRY)}")
        if self.dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset_name {self.dataset_name!r}. Choose from: {list(DATASET_REGISTRY)}")
        if self.output_dir is None:
            self.output_dir = f"{self.model_name}-{self.dataset_name}"

    @property
    def model_spec(self) -> ModelSpec:
        return MODEL_REGISTRY[self.model_name]

    @property
    def dataset_spec(self) -> DatasetSpec:
        return DATASET_REGISTRY[self.dataset_name]


def load_and_split_dataset(cfg: TrainConfig):
    spec = cfg.dataset_spec
    raw_ds = load_dataset(spec.dataset_id)
    full = raw_ds[spec.source_split]
    split = full.train_test_split(test_size=cfg.eval_size, seed=cfg.seed)
    train_ds, eval_ds = split["train"], split["test"]
    if cfg.max_train_samples is not None:
        train_ds = train_ds.select(range(min(cfg.max_train_samples, len(train_ds))))
    if cfg.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(cfg.max_eval_samples, len(eval_ds))))
    return train_ds, eval_ds


def ensure_rgb(dataset):
    for conversation in dataset:
        for message in conversation:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    img = part.get("image")
                    if part.get("type") == "image" and isinstance(img, Image.Image) and img.mode != "RGB":
                        part["image"] = img.convert("RGB")


def load_model_and_processor(cfg: TrainConfig):
    spec = cfg.model_spec
    processor = AutoProcessor.from_pretrained(spec.model_id, **spec.processor_kwargs)
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        spec.model_id,
        device_map="auto",
        **spec.model_kwargs,
    )
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=spec.lora_targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model, processor


def make_collate_fn(processor, max_length: int, collate_style: str):
    _kw = dict(tokenize=True, return_dict=True, return_tensors="pt",
               padding=True, truncation=True, max_length=max_length)

    def collate_chat_template(sample):
        batch = processor.apply_chat_template(sample, **_kw)
        # Prompt-only pass to find where response tokens begin (images included
        # so image token count is accurate).
        prompt_batch = processor.apply_chat_template(
            [convo[:-1] for convo in sample], add_generation_prompt=True, **_kw
        )
        labels = batch["input_ids"].clone()
        for j in range(len(sample)):
            prompt_len = int(prompt_batch["attention_mask"][j].sum())
            labels[j, :prompt_len] = -100
            labels[j, batch["attention_mask"][j] == 0] = -100
        batch["labels"] = labels
        return batch

    def collate_paligemma(sample):
        # PaliGemma uses processor(text, images, suffix) — no chat template.
        # suffix contains the answer; the processor masks prompt tokens with -100 automatically.
        images, prompts, answers = [], [], []
        for convo in sample:
            user_content = next(m["content"] for m in convo if m["role"] == "user")
            images.append(next(p["image"] for p in user_content if p["type"] == "image"))
            prompts.append("<image>\n" + next(p["text"] for p in user_content if p["type"] == "text") + "\n")
            answers.append(next(m["content"][0]["text"] for m in convo if m["role"] == "assistant"))
        return processor(
            text=prompts, images=images, suffix=answers,
            return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        )

    return collate_chat_template if collate_style == "chat_template" else collate_paligemma


class TrainingProgressCallback(TrainerCallback):
    """Prints a clean one-liner per logging step and a summary at the end."""

    def __init__(self):
        self._step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        elapsed = time.time() - self._step_start if self._step_start else 0
        step = state.global_step
        total = state.max_steps
        pct = 100 * step / total if total else 0
        loss = logs["loss"]
        lr = logs.get("learning_rate", 0)
        print(f"  [{step:4d}/{total}  {pct:5.1f}%]  loss={loss:.4f}  lr={lr:.2e}  ({elapsed:.1f}s/step)")


def build_sft_config(cfg: TrainConfig) -> SFTConfig:
    return SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": cfg.min_lr_ratio},
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        optim="adamw_torch_8bit",
        max_length=cfg.max_length,
        dataset_kwargs={"skip_prepare_dataset": True},
        disable_tqdm=True,
        # Eval + checkpoint selection every epoch
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def show_samples(model, processor, conversations: list, output_dir: str, tag: str,
                 collate_style: str = "chat_template", n: int = 3):
    """Run greedy decode on the first n conversations, print results, and save images."""
    from transformers.image_utils import load_image as _load_image

    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    for i, convo in enumerate(conversations[:n]):
        # Strip the assistant turn to get the prompt, keep it for the label
        prompt_turns = convo[:-1]
        label = convo[-1]["content"][0]["text"]

        if collate_style == "paligemma":
            user_content = prompt_turns[-1]["content"]
            image = next(p["image"] for p in user_content if p["type"] == "image")
            question = next(p["text"] for p in user_content if p["type"] == "text")
            inputs = processor(
                text=f"<image>\n{question}", images=image,
                return_tensors="pt",
            ).to(model.device)
        else:
            inputs = processor.apply_chat_template(
                [prompt_turns],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

        # Decode only the newly generated tokens
        generated = processor.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Extract the user prompt text and image for display
        user_content = prompt_turns[-1]["content"]
        user_text = next(p["text"] for p in user_content if p["type"] == "text")
        img_src = next((p["image"] for p in user_content if p["type"] == "image"), None)

        # Save image (load from URL/path if not already a PIL Image)
        if img_src is not None:
            img = img_src if isinstance(img_src, Image.Image) else _load_image(img_src)
            img_path = os.path.join(output_dir, f"sample_{i+1}_{tag}.jpg")
            img.convert("RGB").save(img_path)
            img_note = f"  → {img_path}"
        else:
            img_note = ""

        print(f"  [{i+1}] prompt:    {user_text}{img_note}")
        print(f"       label:     {label}")
        print(f"       generated: {generated}")
    model.train()


class GpuMonitor:
    """Polls nvidia-smi in a background thread; samples are bucketed by phase."""

    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._phase: str = "idle"
        self._samples: dict[str, list[int]] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def set_phase(self, name: str):
        self._phase = name

    def _poll(self):
        while not self._stop.wait(self._interval):
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                )
                val = int(out.decode().strip().splitlines()[0])  # MiB
                self._samples.setdefault(self._phase, []).append(val)
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self._thread.join()

    def peak(self, phase: str) -> float | None:
        s = self._samples.get(phase)
        return max(s) / 1024 if s else None  # MiB → GiB


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def run_one(model_name: str, dataset_name: str = "vqav2") -> dict:
    gpu = GpuMonitor()
    run_start = time.time()
    cfg = TrainConfig(model_name=model_name, dataset_name=dataset_name, use_lora=True)
    spec = cfg.model_spec

    print(f"\n{'='*60}")
    print(f"  Model:   {spec.model_id}")
    print(f"  Dataset: {cfg.dataset_spec.dataset_id}")
    print(f"  Output:  {cfg.output_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()
    print(f"[1/4] Loading model...")
    model, processor = load_model_and_processor(cfg)
    params = model.num_parameters()
    print(f"      {params:,} params  (~{params * 2 / 1e9:.1f} GB bfloat16)  [{_fmt_duration(time.time() - t0)}]\n")

    t0 = time.time()
    print(f"[2/4] Loading dataset...")
    train_raw, eval_raw = load_and_split_dataset(cfg)
    fmt = cfg.dataset_spec.format_fn
    train_dataset = [fmt(s) for s in train_raw]
    eval_dataset = [fmt(s) for s in eval_raw]
    ensure_rgb(train_dataset)
    ensure_rgb(eval_dataset)
    print(f"      train={len(train_dataset):,}  eval={len(eval_dataset):,}  [{_fmt_duration(time.time() - t0)}]")
    print(f"      epochs={cfg.num_train_epochs}  steps/epoch={len(train_dataset) // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)}\n")

    collate_fn = make_collate_fn(processor, cfg.max_length, spec.collate_style)
    sft_config = build_sft_config(cfg)
    processing_class = processor.tokenizer if spec.processing_class == "tokenizer" else processor

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        processing_class=processing_class,
        callbacks=[TrainingProgressCallback()],
    )

    print(f"[3/4] Pre-training eval...")
    t0 = time.time()
    gpu.set_phase("pre_eval")
    pre_metrics = trainer.evaluate()
    pre_loss = pre_metrics.get("eval_loss", float("nan"))
    print(f"      eval_loss={pre_loss:.4f}  [{_fmt_duration(time.time() - t0)}]")
    print(f"      samples (before training):")
    show_samples(model, processor, eval_dataset, cfg.output_dir, tag="before", collate_style=spec.collate_style)
    print()

    print(f"[4/4] Training  ({cfg.num_train_epochs} epoch(s), lr={cfg.learning_rate:.0e}, lora={cfg.use_lora})")
    t0 = time.time()
    gpu.set_phase("train")
    trainer.train()
    train_duration = time.time() - t0
    print(f"      done  [{_fmt_duration(train_duration)}]\n")

    print(f"      Post-training eval...")
    t0 = time.time()
    gpu.set_phase("post_eval")
    post_metrics = trainer.evaluate()
    post_loss = post_metrics.get("eval_loss", float("nan"))
    infer_duration = time.time() - t0
    print(f"      eval_loss={post_loss:.4f}  [{_fmt_duration(infer_duration)}]")
    print(f"      samples (after training):")
    show_samples(model, processor, eval_dataset, cfg.output_dir, tag="after", collate_style=spec.collate_style)

    gpu.stop()

    delta = post_loss - pre_loss
    sign = "-" if delta < 0 else "+"

    def _gpu(phase):
        v = gpu.peak(phase)
        return f"{v:.2f} GB" if v is not None else "n/a"

    print(f"\n{'='*60}")
    print(f"  Model:      {model_name}")
    print(f"  eval_loss:  {pre_loss:.4f}  →  {post_loss:.4f}  ({sign}{abs(delta):.4f})")
    print(f"  train time: {_fmt_duration(train_duration)}  peak_mem={_gpu('train')}")
    print(f"  infer time: {_fmt_duration(infer_duration)}  ({infer_duration / len(eval_dataset) * 1000:.1f} ms/sample)  peak_mem={_gpu('post_eval')}")
    print(f"  total time: {_fmt_duration(time.time() - run_start)}")
    print(f"{'='*60}\n")

    return {"model": model_name, "pre_loss": pre_loss, "post_loss": post_loss,
            "train_time": train_duration, "peak_mem_train": gpu.peak("train")}


def main():
    results = []
    for model_name in MODEL_REGISTRY:
        results.append(run_one(model_name))

    print(f"\n{'='*60}  SUMMARY  {'='*60}")
    print(f"  {'model':<12}  {'pre_loss':>8}  {'post_loss':>9}  {'delta':>7}  {'train_time':>10}  {'peak_mem':>9}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*7}  {'-'*10}  {'-'*9}")
    for r in results:
        delta = r["post_loss"] - r["pre_loss"]
        sign = "-" if delta < 0 else "+"
        mem = f"{r['peak_mem_train']:.2f} GB" if r["peak_mem_train"] else "n/a"
        print(f"  {r['model']:<12}  {r['pre_loss']:>8.4f}  {r['post_loss']:>9.4f}  {sign}{abs(delta):>6.4f}  {_fmt_duration(r['train_time']):>10}  {mem:>9}")
    print(f"{'='*60}{'='*9}{'='*60}\n")


if __name__ == "__main__":
    main()
