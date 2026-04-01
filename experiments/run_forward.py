"""Shared forward, attack, extraction, and PSNR helpers for experiments."""


def run_forward(carrier_t, secret_t, embedder, extractor, feat_ext, enh_pre, device):
    import torch
    import torch.nn.functional as F
    import math
    import io
    import random
    from PIL import Image
    import torchvision.transforms as T

    with torch.no_grad():
        watermarked = embedder(carrier_t, secret_t)
        return watermarked


def apply_attack(watermarked, attack_name):
    import io
    import torch
    from PIL import Image
    import torchvision.transforms as T

    x = watermarked
    if "gaussian" in attack_name.lower():
        sigma = 1 if "1" in attack_name else 10
        return torch.clamp(x + torch.randn_like(x) * (sigma / 255.0), 0, 1)
    elif "jpeg" in attack_name.lower():
        q = 90 if "90" in attack_name else 80
        imgs = []
        for img in x:
            pil = T.ToPILImage()(img.cpu())
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=q)
            buf.seek(0)
            imgs.append(T.ToTensor()(Image.open(buf)))
        return torch.stack(imgs).to(x.device)
    else:
        return (x * 255).round() / 255.0


def extract_watermark(watermarked, attacked, feat_ext, enh_pre, extractor):
    import torch

    with torch.no_grad():
        diff = feat_ext(watermarked, attacked)
        enhanced = enh_pre(attacked)
        extracted = extractor(enhanced, diff)
    return extracted


def psnr_val(a, b):
    import math
    import torch

    mse = torch.nn.functional.mse_loss(a, b).item()
    return 100.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)
