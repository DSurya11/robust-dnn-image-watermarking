"""Loss functions for the watermarking pipeline."""

from typing import Dict

import torch
import torch.nn.functional as F


EPS = 1e-8


def ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Differentiable SSIM loss implemented in pure PyTorch.
    """
    window_size = 11
    c1, c2 = 0.01**2, 0.03**2

    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    kernel = kernel / kernel.sum()
    kernel = kernel.expand(pred.shape[1], 1, window_size, window_size)

    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=pred.shape[1])
    mu2 = F.conv2d(target, kernel, padding=pad, groups=pred.shape[1])

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=pred.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=pred.shape[1]) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=pred.shape[1]) - mu12

    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return 1.0 - ssim_map.mean()


def _haar_ll(x: torch.Tensor) -> torch.Tensor:
    """Returns the LL sub-band from a single-level 2D Haar DWT."""
    if x.dim() != 4:
        raise ValueError("Expected input with shape (B, C, H, W).")

    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]

    ll = (x00 + x01 + x10 + x11) * 0.5
    return ll


def _safe_log(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps, max=1.0 - eps))


def compute_losses(
    carrier_image: torch.Tensor,
    secret_image: torch.Tensor,
    watermarked_image: torch.Tensor,
    attacked_image: torch.Tensor,
    extracted_image: torch.Tensor,
    d_carrier: torch.Tensor,
    d_attacked: torch.Tensor,
    lambda_c: float = 1.0,
    lambda_s: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Computes all requested losses and returns them as a dictionary."""

    l_pre = F.mse_loss(watermarked_image, carrier_image)
    l_post = F.mse_loss(extracted_image, secret_image)
    l_enhance = lambda_c * l_pre + lambda_s * l_post

    # Discriminator loss: L_D = -E[log(D(carrier))] - E[log(1 - D(attacked))]
    l_d = -torch.mean(_safe_log(d_carrier)) - torch.mean(_safe_log(1.0 - d_attacked))

    # Generator adversarial loss: L_G = -E[log(D(attacked))]
    l_g = -torch.mean(_safe_log(d_attacked))

    # Adversarial regularizer: L_adv = E[log(D(carrier))] + E[log(1 - D(attacked))]
    l_adv = torch.mean(_safe_log(d_carrier)) + torch.mean(_safe_log(1.0 - d_attacked))

    l_f = F.mse_loss(watermarked_image, attacked_image)

    ll_watermarked = _haar_ll(watermarked_image)
    ll_carrier = _haar_ll(carrier_image)
    l_wavelet = F.mse_loss(ll_watermarked, ll_carrier)

    l_stage = l_enhance + 0.1 * l_adv + 0.05 * l_f + l_wavelet

    return {
        "L_pre": l_pre,
        "L_post": l_post,
        "L_enhance": l_enhance,
        "L_D": l_d,
        "L_G": l_g,
        "L_adv": l_adv,
        "L_f": l_f,
        "L_wavelet": l_wavelet,
        "L_stage": l_stage,
    }
