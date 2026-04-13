"""Forward helpers for running attacks during evaluation.

This file exists because multiple scripts import `experiments.run_forward.apply_attack`.
It intentionally stays lightweight and depends only on `utils.attacks`.
"""

from __future__ import annotations

import torch

from utils.attacks import gaussian_noise, jpeg_compress, round_error


def apply_attack(x: torch.Tensor, attack_key: str) -> torch.Tensor:
    """Apply a named attack to a BCHW (or CHW) tensor in [0,1]."""
    key = str(attack_key).strip()

    if key in {"Gaussian_s1", "gaussian_s1"}:
        return gaussian_noise(x, sigma=1)
    if key in {"Gaussian_s10", "gaussian_s10"}:
        return gaussian_noise(x, sigma=10)
    if key in {"JPEG_q90", "jpeg_q90"}:
        return jpeg_compress(x, quality=90)
    if key in {"JPEG_q80", "jpeg_q80"}:
        return jpeg_compress(x, quality=80)
    if key in {"Round", "round"}:
        return round_error(x)

    raise ValueError(
        f"Unknown attack_key={attack_key!r}. Expected one of: Gaussian_s1, Gaussian_s10, JPEG_q90, JPEG_q80, Round."
    )
