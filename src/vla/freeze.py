"""Selective freezing of the VLM backbone.

Modes:
  none      — full fine-tune (no-op).
  vision    — freeze vision tower + connector (if any).
  language  — freeze language model + lm_head.
  all       — freeze every group above.

The action head (anything outside `vla.vlm`) is never touched here, including
the learnable `action_query` embedding owned by the MLP VLA — it stays
trainable in every mode.
"""

from dataclasses import dataclass, field

import torch.nn as nn


FREEZE_MODES = ("none", "vision", "language", "all")


@dataclass
class FreezeReport:
    mode: str
    frozen_paths: list[str] = field(default_factory=list)  # dotted paths under vla.vlm
    skipped_paths: list[str] = field(default_factory=list)  # paths missing from this VLM


def _resolve(root: nn.Module, dotted: str) -> nn.Module | None:
    obj = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj if isinstance(obj, nn.Module) else None


def _groups_to_freeze(mode: str) -> tuple[str, ...]:
    if mode == "none":
        return ()
    if mode == "vision":
        return ("vision", "connector")
    if mode == "language":
        return ("language",)
    if mode == "all":
        return ("vision", "connector", "language")
    raise ValueError(f"Unknown freeze mode: {mode!r}; expected one of {FREEZE_MODES}")


def apply_freeze(vla: nn.Module, mode: str, freeze_groups: dict) -> FreezeReport:
    """Freeze the requested submodules in-place. Returns a report."""
    report = FreezeReport(mode=mode)
    if mode == "none":
        return report

    for group in _groups_to_freeze(mode):
        for path in freeze_groups.get(group, ()):
            mod = _resolve(vla.vlm, path)
            if mod is None:
                report.skipped_paths.append(path)
                continue
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()
            report.frozen_paths.append(path)

    return report


def reapply_eval(vla: nn.Module, mode: str, freeze_groups: dict) -> None:
    """Re-put frozen submodules into eval() after a vla.train() call.

    Keeps dropout / layernorm-stats deterministic in the frozen pathway.
    """
    if mode == "none":
        return
    for group in _groups_to_freeze(mode):
        for path in freeze_groups.get(group, ()):
            mod = _resolve(vla.vlm, path)
            if mod is not None:
                mod.eval()
