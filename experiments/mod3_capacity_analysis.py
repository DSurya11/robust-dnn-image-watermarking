"""MOD 3: Capacity analysis with programmatically generated secret types at 128x128."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from load_models import load_all_models
from run_forward import run_forward, extract_watermark, psnr_val, apply_attack


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")


def tensor_to_hwc_np(x: torch.Tensor):
    return x.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()


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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder, extractor, _disc, feat_ext, enh_pre, _enh_post = load_all_models(device)

    carrier_path = DATA_DIR / "sample_0.jpg"
    natural_path = DATA_DIR / "sample_2.jpg"
    if not natural_path.exists():
        files = sorted(DATA_DIR.glob("*.jpg")) + sorted(DATA_DIR.glob("*.png"))
        files = [f for f in files if "prepare" not in f.name]
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
    for name, secret_t in secrets.items():
        watermarked = run_forward(carrier_t, secret_t, embedder, extractor, feat_ext, enh_pre, device)
        attacked = apply_attack(watermarked, "gaussian10")
        extracted = extract_watermark(watermarked, attacked, feat_ext, enh_pre, extractor)

        psnr_c = psnr_val(watermarked, carrier_t)
        psnr_s = psnr_val(extracted, secret_t)
        ssim = structural_similarity(
            tensor_to_hwc_np(secret_t),
            tensor_to_hwc_np(extracted),
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
    plt.title("MOD 3: Secret Complexity vs Extraction Quality")
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
