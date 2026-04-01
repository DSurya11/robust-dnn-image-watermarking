import argparse
import csv
import io
import math
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def tensor_to_numpy(t):
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)


def numpy_to_tensor(arr, device):
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)


def load_image(path, size=128, device="cpu"):
    img = Image.open(path).convert("RGB").resize((size, size))
    return T.ToTensor()(img).unsqueeze(0).to(device)


def psnr(a, b):
    mse = torch.nn.functional.mse_loss(a, b).item()
    return 100.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)


def resolve_checkpoint(requested="models/checkpoints/best.pth"):
    candidates = [requested, "models/checkpoints/best.pth", "models/checkpoints/latest.pth"]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists():
            print(f"Using checkpoint: {candidate}")
            return candidate
    print("WARNING: No checkpoint found (best.pth/latest.pth). Using random weights.")
    return "models/checkpoints/latest.pth"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--secret", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from experiments.load_models import load_all_models
    from experiments.run_forward import apply_attack
    from utils.visualize import visualize_pipeline, plot_psnr_comparison, plot_modification_results

    checkpoint_path = resolve_checkpoint("models/checkpoints/best.pth")
    embedder, extractor, disc, feat_ext, enh_pre, enh_post = load_all_models(device, checkpoint_path)

    carrier = load_image(args.carrier, size=128, device=device)
    secret = load_image(args.secret, size=128, device=device)

    attacks = ["Gaussian_s1", "Gaussian_s10", "JPEG_q90", "JPEG_q80", "Round"]
    results = []

    for attack_name in attacks:
        with torch.no_grad():
            watermarked = embedder(carrier, secret)
            attacked = apply_attack(watermarked, attack_name)
            diff = feat_ext(watermarked, attacked)
            enhanced = enh_pre(attacked)
            extracted = extractor(enhanced, diff)

            psnr_c = psnr(carrier, watermarked)
            psnr_s = psnr(secret, extracted)
            result = "PASS" if psnr_s > 20 else "FAIL"

            visualize_pipeline(
                carrier=tensor_to_numpy(carrier),
                secret=tensor_to_numpy(secret),
                watermarked=tensor_to_numpy(watermarked),
                attacked=tensor_to_numpy(attacked),
                extracted=tensor_to_numpy(extracted),
                attack_name=attack_name,
                save_path=f"results/visual_{attack_name}.png",
            )

            results.append(
                {
                    "attack": attack_name,
                    "psnr_c": psnr_c,
                    "psnr_s": psnr_s,
                    "result": result,
                }
            )

    print("=" * 58)
    print(f"{'Attack':<18}|{'PSNR-C':>8}|{'PSNR-S':>8}| Result")
    print("-" * 58)
    for r in results:
        print(f"{r['attack']:<18}|{r['psnr_c']:>8.2f}|{r['psnr_s']:>8.2f}| {r['result']}")
    print("=" * 58)

    os.makedirs("results", exist_ok=True)
    csv_path = "results/evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "psnr_c", "psnr_s", "result"])
        writer.writeheader()
        writer.writerows(results)

    plot_psnr_comparison("results/evaluation_results.csv")
    plot_modification_results()

    print("All visual outputs saved to results/")
    print("Report-ready summary chart: results/all_modifications_summary.png")


if __name__ == "__main__":
    main()
