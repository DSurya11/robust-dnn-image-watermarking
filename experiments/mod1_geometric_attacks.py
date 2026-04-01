"""MOD 1: Geometric attack robustness using shared experiment helpers."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from load_models import load_all_models
from run_forward import run_forward, extract_watermark, psnr_val, apply_attack


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")


def load_pairs(num_pairs: int = 8, size: int = 128) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    files = sorted(DATA_DIR.glob("*.jpg")) + sorted(DATA_DIR.glob("*.png"))
    files = [f for f in files if "prepare" not in f.name]
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])

    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    max_pairs = min(num_pairs, max(1, len(files) - 1))
    for i in range(max_pairs):
        c = tfm(Image.open(files[i]).convert("RGB"))
        s = tfm(Image.open(files[(i + 1) % len(files)]).convert("RGB"))
        pairs.append((c, s))
    return pairs


def geometric_attack(x: torch.Tensor, attack_name: str, intensity: float) -> torch.Tensor:
    if attack_name == "Rotation":
        out = [TF.rotate(img, float(intensity), interpolation=T.InterpolationMode.BILINEAR, fill=0.0) for img in x]
        return torch.stack(out, dim=0)
    if attack_name == "Crop":
        out = []
        for img in x:
            h, w = img.shape[1:]
            ch = max(1, int(round(h * intensity)))
            cw = max(1, int(round(w * intensity)))
            cropped = TF.center_crop(img, [ch, cw])
            out.append(TF.resize(cropped, [h, w], interpolation=T.InterpolationMode.BILINEAR, antialias=True))
        return torch.stack(out, dim=0)
    if attack_name == "Brightness":
        return torch.clamp(x * float(intensity), 0, 1)
    return apply_attack(x, "round")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder, extractor, _disc, feat_ext, enh_pre, _enh_post = load_all_models(device)
    pairs = load_pairs(num_pairs=8, size=128)

    attack_grid: Dict[str, List[Tuple[str, float]]] = {
        "Rotation": [("mild", 5), ("medium", 15), ("strong", 30)],
        "Crop": [("mild", 0.95), ("medium", 0.88), ("strong", 0.80)],
        "Brightness": [("mild", 0.8), ("medium", 1.2), ("strong", 1.4)],
    }

    rows: List[Dict[str, float | str]] = []
    summary: Dict[str, Dict[str, float]] = {k: {} for k in attack_grid}

    for attack_name, levels in attack_grid.items():
        for level_name, intensity in levels:
            vals = []
            for carrier, secret in pairs:
                carrier_t = carrier.unsqueeze(0).to(device)
                secret_t = secret.unsqueeze(0).to(device)
                watermarked = run_forward(carrier_t, secret_t, embedder, extractor, feat_ext, enh_pre, device)
                attacked = geometric_attack(watermarked, attack_name, intensity)
                extracted = extract_watermark(watermarked, attacked, feat_ext, enh_pre, extractor)
                vals.append(psnr_val(extracted, secret_t))

            mean_psnr = float(sum(vals) / max(1, len(vals)))
            summary[attack_name][level_name] = mean_psnr
            rows.append(
                {
                    "Attack": attack_name,
                    "Level": level_name,
                    "Intensity": intensity,
                    "PSNR-S": mean_psnr,
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "mod1_geometric_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Attack", "Level", "Intensity", "PSNR-S"])
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(10, 5))
    for attack_name in ["Rotation", "Crop", "Brightness"]:
        subset = [r for r in rows if r["Attack"] == attack_name]
        x_vals = [str(r["Intensity"]) for r in subset]
        y_vals = [float(r["PSNR-S"]) for r in subset]
        plt.plot(x_vals, y_vals, marker="o", label=attack_name)
    plt.title("MOD 1: Geometric Attack Robustness")
    plt.xlabel("Attack Intensity")
    plt.ylabel("PSNR-S (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    chart_path = RESULTS_DIR / "mod1_geometric_chart.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print("Attack           | Mild    | Medium  | Strong")
    for attack_name in ["Rotation", "Crop", "Brightness"]:
        print(
            f"{attack_name:<16} | "
            f"{summary[attack_name]['mild']:.2f}   | "
            f"{summary[attack_name]['medium']:.2f}   | "
            f"{summary[attack_name]['strong']:.2f}"
        )
    print(f"Saved: {csv_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
