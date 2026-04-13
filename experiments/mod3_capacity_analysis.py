"""MOD 3: Capacity analysis with generated secret types for SimpleISN."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluate import select_best_checkpoint
from run_forward import apply_attack


RESULTS_DIR = Path("results/final")
DATA_DIR = Path("data")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    return 100.0 if mse < 1e-10 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def tensor_to_hwc_np(x: torch.Tensor):
    return x.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()


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


def load_image(path: Path, size: int = 128) -> torch.Tensor:
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    return tfm(Image.open(path).convert("RGB")).unsqueeze(0)


def make_text_secret(size: int = 128) -> torch.Tensor:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((12, 52), "COPYRIGHT", fill=(255, 255, 255))
    return T.ToTensor()(img).unsqueeze(0)


def make_logo_secret(size: int = 128) -> torch.Tensor:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    c = size // 2
    for r in [14, 26, 38]:
        draw.ellipse((c - r, c - r, c + r, c + r), outline=(255, 255, 255), width=2)
    return T.ToTensor()(img).unsqueeze(0)


def get_clean_psnr(model_path: str) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isn = load_isn(device, model_path)

    files = sorted(DATA_DIR.glob("*.jpg")) + sorted(DATA_DIR.glob("*.png"))
    files = [f for f in files if "prepare" not in f.name]
    if len(files) < 2:
        return 0.0

    xh = load_image(files[0], 128).to(device)
    xs = load_image(files[1], 128).to(device)
    with torch.no_grad():
        xc = isn.embed(xh, xs)
        xe = isn.extract(xc)
        return psnr(xe, xs)


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(DATA_DIR.glob("*.jpg")) + sorted(DATA_DIR.glob("*.png"))
    files = [f for f in files if "prepare" not in f.name]
    if len(files) < 2:
        print("ERROR: Model not trained properly")
        sys.exit(1)

    selection_pairs = [
        (load_image(files[i], 128), load_image(files[(i + 1) % len(files)], 128))
        for i in range(min(3, max(1, len(files) - 1)))
    ]
    model_path, _ = select_best_checkpoint(device, selection_pairs)
    if model_path is None:
        print("ERROR: Model not trained properly")
        sys.exit(1)

    clean_psnr = get_clean_psnr(model_path)
    print("=== Running Modification 3 ===")
    print(f"Model checkpoint: {Path(model_path).name}")
    print(f"Sanity check PSNR-S: {clean_psnr:.2f} dB")
    if clean_psnr < 17:
        print("ERROR: Model not trained properly")
        sys.exit(1)
    print("Proceeding...")

    isn = load_isn(device, model_path)

    carrier_path = DATA_DIR / "sample_0.jpg"
    natural_path = DATA_DIR / "sample_2.jpg"
    if not natural_path.exists():
        files = sorted(DATA_DIR.glob("*.jpg")) + sorted(DATA_DIR.glob("*.png"))
        files = [f for f in files if "prepare" not in f.name]
        if not files:
            print("ERROR: No input images found in data/.")
            sys.exit(1)
        natural_path = files[0]

    carrier_t = load_image(carrier_path, size=128).to(device)
    natural_secret = load_image(natural_path, size=128).to(device)

    secrets = {
        "Noise": torch.rand(1, 3, 128, 128, device=device),
        "Text": make_text_secret(128).to(device),
        "Logo": make_logo_secret(128).to(device),
        "Natural": natural_secret,
    }

    rows = []
    with torch.no_grad():
        for name, secret_t in secrets.items():
            xc = isn.embed(carrier_t, secret_t)
            xd = apply_attack(xc, "Gaussian_s10")
            xe = isn.extract(xd)

            psnr_c = psnr(xc, carrier_t)
            psnr_s = psnr(xe, secret_t)
            ssim = structural_similarity(
                tensor_to_hwc_np(secret_t),
                tensor_to_hwc_np(xe),
                channel_axis=2,
                data_range=1.0,
            )
            rows.append(
                {
                    "Secret Type": name,
                    "PSNR-C": float(psnr_c),
                    "PSNR-S": float(psnr_s),
                    "SSIM": float(ssim),
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "mod3_capacity_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Secret Type", "PSNR-C", "PSNR-S", "SSIM"])
        writer.writeheader()
        writer.writerows(rows)

    labels = [r["Secret Type"] for r in rows]
    psnr_s = [r["PSNR-S"] for r in rows]
    ssim_vals = [r["SSIM"] for r in rows]

    x = list(range(len(labels)))
    w = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar([i - w / 2 for i in x], psnr_s, width=w, label="PSNR-S")
    plt.bar([i + w / 2 for i in x], ssim_vals, width=w, label="SSIM")
    plt.xticks(x, labels)
    plt.title("MOD 3: Secret Complexity vs Extraction Quality (v2)")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    chart_path = RESULTS_DIR / "mod3_capacity_chart.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print("Secret Type | PSNR-C | PSNR-S | SSIM")
    for row in rows:
        print(
            f"{row['Secret Type']:<10} | {row['PSNR-C']:.2f} | "
            f"{row['PSNR-S']:.2f} | {row['SSIM']:.3f}"
        )
    print(f"Saved: {csv_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
