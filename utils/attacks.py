"""Image attack utilities for watermark robustness testing."""

import random
from typing import Callable, List

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


Tensor = torch.Tensor


def _validate_tensor(image_tensor: Tensor) -> None:
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a PyTorch tensor.")
    if image_tensor.dim() not in (3, 4):
        raise ValueError("image_tensor must have shape (C,H,W) or (B,C,H,W).")
    if image_tensor.dim() == 3 and image_tensor.size(0) != 3:
        raise ValueError("For 3D input, expected shape (3,H,W).")
    if image_tensor.dim() == 4 and image_tensor.size(1) != 3:
        raise ValueError("For 4D input, expected shape (B,3,H,W).")


def _apply_per_image(image_tensor: Tensor, fn: Callable[[Tensor], Tensor]) -> Tensor:
    """Applies an image transform to either CHW or BCHW tensor input."""
    _validate_tensor(image_tensor)

    if image_tensor.dim() == 3:
        return fn(image_tensor).clamp(0.0, 1.0)

    attacked = [fn(img).clamp(0.0, 1.0) for img in image_tensor]
    return torch.stack(attacked, dim=0)


def gaussian_noise(image_tensor: Tensor, sigma: float = 10) -> Tensor:
    """Adds Gaussian noise with standard deviation sigma in 8-bit intensity units."""

    def _noise_fn(img: Tensor) -> Tensor:
        noise = torch.randn_like(img) * (sigma / 255.0)
        return img + noise

    return _apply_per_image(image_tensor, _noise_fn)


def jpeg_compress(image_tensor: Tensor, quality: int = 80) -> Tensor:
    """Applies JPEG compression via OpenCV encode/decode and returns a tensor in [0,1]."""
    quality = int(max(1, min(100, quality)))

    def _jpeg_fn(img: Tensor) -> Tensor:
        img_cpu = img.detach().clamp(0.0, 1.0).cpu()
        img_np = (img_cpu.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)

        # OpenCV uses BGR order for image codecs.
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(
            ".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            raise RuntimeError("OpenCV JPEG encoding failed.")

        decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded_bgr is None:
            raise RuntimeError("OpenCV JPEG decoding failed.")

        decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        decoded_tensor = torch.from_numpy(decoded_rgb).permute(2, 0, 1).float() / 255.0
        return decoded_tensor.to(img.device)

    return _apply_per_image(image_tensor, _jpeg_fn)


def round_error(image_tensor: Tensor) -> Tensor:
    """Simulates quantization by converting float tensor to uint8 and back to float."""

    def _round_fn(img: Tensor) -> Tensor:
        uint8_img = (img.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        return uint8_img.float() / 255.0

    return _apply_per_image(image_tensor, _round_fn)


def rotation_attack(image_tensor: Tensor, angle: float = 15) -> Tensor:
    """Rotate and inverse-rotate to mimic real-world alignment distortions."""

    def _rot_fn(img: Tensor) -> Tensor:
        rotated = TF.rotate(
            img,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0,
        )
        restored = TF.rotate(
            rotated,
            angle=-angle,
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0,
        )
        return restored

    return _apply_per_image(image_tensor, _rot_fn)


def crop_attack(image_tensor: Tensor, crop_ratio: float = 0.9) -> Tensor:
    """Center-crop by ratio then resize back to the original resolution."""
    crop_ratio = float(max(0.1, min(1.0, crop_ratio)))

    def _crop_fn(img: Tensor) -> Tensor:
        _, h, w = img.shape
        crop_h = max(1, int(round(h * crop_ratio)))
        crop_w = max(1, int(round(w * crop_ratio)))
        cropped = TF.center_crop(img, [crop_h, crop_w])
        resized = TF.resize(
            cropped,
            [h, w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return resized

    return _apply_per_image(image_tensor, _crop_fn)


def brightness_attack(image_tensor: Tensor, factor: float = 1.3) -> Tensor:
    """Scale brightness by a multiplicative factor."""

    def _brightness_fn(img: Tensor) -> Tensor:
        return img * float(factor)

    return _apply_per_image(image_tensor, _brightness_fn)


def apply_random_attack(image_tensor: Tensor) -> Tensor:
    """Randomly applies one supported attack and returns the attacked tensor."""
    attacks: List[Callable[[Tensor], Tensor]] = [
        lambda x: gaussian_noise(x, sigma=10),
        lambda x: jpeg_compress(x, quality=80),
        round_error,
    ]

    attack_fn = random.choice(attacks)
    attacked = attack_fn(image_tensor)
    return attacked.clamp(0.0, 1.0)
