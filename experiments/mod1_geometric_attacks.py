"""MOD 1: Geometric attack robustness for SimpleISN."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return 100.0 if mse < 1e-10 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def resolve_checkpoint(requested: str = "models/checkpoints/phase3_final.pth") -> str | None:
    candidates = [
        requested,
        "models/checkpoints/phase3_final.pth",
        "models/checkpoints/phase2_best.pth",
    ]
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists():
            return candidate
    return None


def load_isn(device: torch.device, model_path: str):
    from models.generator import SimpleISN

    isn = SimpleISN().to(device)
    ckpt = torch.load(model_path, map_location=device)
    if "isn_state_dict" in ckpt:
        isn.load_state_dict(ckpt["isn_state_dict"])
    elif "isn" in ckpt:
        isn.load_state_dict(ckpt["isn"])
    elif "generator" in ckpt:
        isn.load_state_dict(ckpt["generator"])
    else:
        raise KeyError("Checkpoint missing ISN weights.")
    isn.eval()
    return isn


def load_pairs(num_pairs: int = 8, size: int = 128) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    files = sorted(DATA_DIR.glob("*.jpg")) + sorted(DATA_DIR.glob("*.png"))
    files = [f for f in files if "prepare" not in f.name]
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])

    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    max_pairs = min(num_pairs, max(1, len(files) - 1))
    for i in range(max_pairs):
        c = tfm(Image.open(files[i]).convert("RGB"))
        s = tfm(Image.open(files[(i + 1) % len(files)]).convert("RGB"))
        pairs.append((c.unsqueeze(0), s.unsqueeze(0)))
    return pairs


def get_clean_psnr(model_path: str) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isn = load_isn(device, model_path)
    pairs = load_pairs(num_pairs=1, size=128)
    if not pairs:
        return 0.0
    with torch.no_grad():
        xh, xs = pairs[0]
        xh = xh.to(device)
        xs = xs.to(device)
        xc = isn.embed(xh, xs)
        xe = isn.extract(xc)
        return psnr(xe, xs)


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
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/phase3_final.pth")
    args = parser.parse_args()

    model_path = resolve_checkpoint(args.checkpoint)
    if model_path is None:
        print("ERROR: Model not trained properly. Run train.py first.")
        sys.exit(1)

    clean_psnr = get_clean_psnr(model_path)
    print("=== Running Modification 1 ===")
    print(f"Model checkpoint: {Path(model_path).name}")
    print(f"Sanity check PSNR-S: {clean_psnr:.2f} dB")
    print("Proceeding...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isn = load_isn(device, model_path)
    pairs = load_pairs(num_pairs=8, size=128)

    attack_grid: Dict[str, List[Tuple[str, float]]] = {
        "Rotation": [("mild", 5), ("medium", 15), ("strong", 30)],
        "Crop": [("mild", 0.95), ("medium", 0.88), ("strong", 0.80)],
        "Brightness": [("mild", 0.8), ("medium", 1.2), ("strong", 1.4)],
    }

    rows: List[Dict[str, float | str]] = []
    summary: Dict[str, Dict[str, float]] = {k: {} for k in attack_grid}

    with torch.no_grad():
        for attack_name, levels in attack_grid.items():
            for level_name, intensity in levels:
                vals = []
                for xh, xs in pairs:
                    xh = xh.to(device)
                    xs = xs.to(device)
                    xc = isn.embed(xh, xs)
                    xd = geometric_attack(xc, attack_name, intensity)
                    xe = isn.extract(xd)
                    vals.append(psnr(xe, xs))

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
    csv_path = RESULTS_DIR / "mod1_geometric_results_v2.csv"
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
    plt.title("MOD 1: Geometric Attack Robustness (v2)")
    plt.xlabel("Attack Intensity")
    plt.ylabel("PSNR-S (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    chart_path = RESULTS_DIR / "mod1_geometric_chart_v2.png"
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
