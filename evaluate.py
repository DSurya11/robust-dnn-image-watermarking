"""End-to-end checkpoint evaluation over all image pairs in data folder."""

import csv
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from experiments.load_models import load_all_models
from experiments.run_forward import apply_attack
from utils.visualize import plot_psnr_comparison


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return 100.0 if mse < 1e-10 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def load_pairs(data_folder: str, size: int = 128):
    folder = Path(data_folder)
    files = sorted(folder.glob("*.jpg"))
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    pairs = []
    for i in range(max(1, len(files) - 1)):
        c = tfm(Image.open(files[i]).convert("RGB"))
        s = tfm(Image.open(files[(i + 1) % len(files)]).convert("RGB"))
        pairs.append((c.unsqueeze(0), s.unsqueeze(0)))
    return pairs


def resolve_checkpoint(requested: str = "models/checkpoints/best.pth") -> str | None:
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
    return None


def evaluate_model(checkpoint="models/checkpoints/best.pth", data_folder="data/", results_folder="results/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = resolve_checkpoint(checkpoint)
    embedder, extractor, _disc, feat_ext, enh_pre, _enh_post = load_all_models(
        device,
        resolved or "models/checkpoints/latest.pth",
    )

    pairs = load_pairs(data_folder, size=128)
    attacks = [
        ("Gaussian s=1", "Gaussian_s1"),
        ("Gaussian s=10", "Gaussian_s10"),
        ("JPEG q=90", "JPEG_q90"),
        ("JPEG q=80", "JPEG_q80"),
        ("Round", "Round"),
    ]

    stats = {name: {"psnr_c": 0.0, "psnr_s": 0.0, "count": 0} for name, _ in attacks}

    with torch.no_grad():
        for carrier, secret in pairs:
            carrier = carrier.to(device)
            secret = secret.to(device)
            watermarked = embedder(carrier, secret)

            for print_name, attack_key in attacks:
                attacked = apply_attack(watermarked, attack_key)
                diff = feat_ext(watermarked, attacked)
                enhanced = enh_pre(attacked)
                extracted = extractor(enhanced, diff)

                stats[print_name]["psnr_c"] += psnr(carrier, watermarked)
                stats[print_name]["psnr_s"] += psnr(secret, extracted)
                stats[print_name]["count"] += 1

    rows = []
    for print_name, _ in attacks:
        c = max(1, stats[print_name]["count"])
        rows.append(
            {
                "attack": print_name,
                "psnr_c": stats[print_name]["psnr_c"] / c,
                "psnr_s": stats[print_name]["psnr_s"] / c,
            }
        )

    print("Attack           | PSNR-C | PSNR-S")
    for r in rows:
        print(f"{r['attack']:<16} | {r['psnr_c']:.2f}  | {r['psnr_s']:.2f}")

    results_path = Path(results_folder)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "evaluation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "psnr_c", "psnr_s"])
        writer.writeheader()
        writer.writerows(rows)

    plot_psnr_comparison(str(csv_path))
    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate_model()
