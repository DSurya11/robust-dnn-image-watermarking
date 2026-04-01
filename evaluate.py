"""Evaluation for Phase 1 (with fallback to Phase 2) over all image pairs in data/."""

import argparse
import csv
import math
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from experiments.run_forward import apply_attack


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.nn.functional.mse_loss(a, b).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def load_pairs(data_folder: str, size: int = 128):
    folder = Path(data_folder)
    files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])

    pairs = []
    for i in range(max(1, len(files) - 1)):
        carrier = tfm(Image.open(files[i]).convert("RGB"))
        secret = tfm(Image.open(files[(i + 1) % len(files)]).convert("RGB"))
        pairs.append((carrier.unsqueeze(0), secret.unsqueeze(0)))
    return pairs


def resolve_checkpoint(requested: str) -> str | None:
    requested_path = Path(requested)
    phase1 = Path("models/checkpoints/phase1_best.pth")
    phase2 = Path("models/checkpoints/phase2_best.pth")

    candidates = [requested_path, phase1, phase2]
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            print(f"Using checkpoint: {candidate}")
            return str(candidate)

    print("WARNING: No valid checkpoint found (phase1_best.pth or phase2_best.pth).")
    return None


def _load_models_for_eval(device: torch.device, checkpoint: str):
    from models.enhance import EnhanceModule
    from models.feat_extract import DifferentialFeatureExtractor
    from models.generator import SimpleISN

    ckpt = torch.load(checkpoint, map_location=device)

    isn = SimpleISN().to(device)
    isn_key = "isn_state_dict" if "isn_state_dict" in ckpt else "isn"
    if isn_key not in ckpt:
        raise KeyError("Checkpoint missing ISN weights.")
    isn.load_state_dict(ckpt[isn_key])
    isn.eval()

    use_phase3_stack = (
        "enhance_pre_state_dict" in ckpt
        and "enhance_post_state_dict" in ckpt
        and "feat_extract_state_dict" in ckpt
    )

    enhance_pre = None
    enhance_post = None
    feat_extract = None
    if use_phase3_stack:
        enhance_pre = EnhanceModule().to(device)
        enhance_post = EnhanceModule().to(device)
        feat_extract = DifferentialFeatureExtractor().to(device)

        enhance_pre.load_state_dict(ckpt["enhance_pre_state_dict"])
        enhance_post.load_state_dict(ckpt["enhance_post_state_dict"])
        feat_extract.load_state_dict(ckpt["feat_extract_state_dict"])

        enhance_pre.eval()
        enhance_post.eval()
        feat_extract.eval()

    return isn, enhance_pre, enhance_post, feat_extract


def extract_with_model(
    isn,
    watermarked: torch.Tensor,
    attacked: torch.Tensor,
    enhance_pre,
    enhance_post,
    feat_extract,
) -> torch.Tensor:
    if enhance_pre is None or enhance_post is None or feat_extract is None:
        return isn.extract(attacked)

    diff_feat = feat_extract(watermarked, attacked)
    attacked_enhanced = enhance_pre(attacked + diff_feat)
    extracted = isn.extract(attacked_enhanced)
    return enhance_post(extracted)


def evaluate(checkpoint: str, data_folder: str = "data/", results_folder: str = "results_phase1/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = resolve_checkpoint(checkpoint)
    if resolved is None:
        return

    isn, enhance_pre, enhance_post, feat_extract = _load_models_for_eval(device, resolved)

    pairs = load_pairs(data_folder, size=128)
    if len(pairs) == 0:
        print("No image pairs found in data folder.")
        return

    # Required sanity check on one image pair with clean extraction.
    with torch.no_grad():
        xh0, xs0 = pairs[0]
        xh0 = xh0.to(device)
        xs0 = xs0.to(device)
        xc0 = isn.embed(xh0, xs0)
        xe0 = extract_with_model(isn, xc0, xc0, enhance_pre, enhance_post, feat_extract)
        clean_psnr_s = psnr(xs0, xe0)
        print(f"Sanity check - Clean PSNR-S: {clean_psnr_s:.2f} dB")
        if clean_psnr_s < 18.0:
            print("WARNING: Model undertrained. Results will be poor.")

    attacks = [
        ("Gaussian_s1", "Gaussian_s1"),
        ("Gaussian_s10", "Gaussian_s10"),
        ("JPEG_q90", "JPEG_q90"),
        ("JPEG_q80", "JPEG_q80"),
        ("Round", "Round"),
    ]
    stats = {name: {"psnr_c": 0.0, "psnr_s": 0.0, "count": 0} for name, _ in attacks}

    with torch.no_grad():
        for xh, xs in pairs:
            xh = xh.to(device)
            xs = xs.to(device)
            xc = isn.embed(xh, xs)
            psnr_c_val = psnr(xh, xc)

            for name, attack_key in attacks:
                xd = apply_attack(xc, attack_key)
                xe = extract_with_model(isn, xc, xd, enhance_pre, enhance_post, feat_extract)

                stats[name]["psnr_c"] += psnr_c_val
                stats[name]["psnr_s"] += psnr(xs, xe)
                stats[name]["count"] += 1

    rows = []
    for name, _ in attacks:
        count = max(1, stats[name]["count"])
        avg_c = stats[name]["psnr_c"] / count
        avg_s = stats[name]["psnr_s"] / count
        rows.append(
            {
                "attack": name,
                "psnr_c": avg_c,
                "psnr_s": avg_s,
                "result": "PASS" if avg_s > 18.0 else "FAIL",
            }
        )

    print("=" * 58)
    print("Attack            |  PSNR-C|  PSNR-S| Result")
    print("-" * 58)
    for row in rows:
        print(f"{row['attack']:<17} | {row['psnr_c']:7.2f}| {row['psnr_s']:7.2f}| {row['result']}")
    print("=" * 58)
    print("PASS threshold: PSNR-S > 18 dB")

    results_path = Path(results_folder)
    results_path.mkdir(parents=True, exist_ok=True)
    csv_path = results_path / "evaluation_results_phase1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["attack", "psnr_c", "psnr_s", "result"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/phase1_best.pth")
    parser.add_argument("--data_folder", type=str, default="data/")
    parser.add_argument("--results_folder", type=str, default="results_phase1/")
    args = parser.parse_args()
    evaluate(checkpoint=args.checkpoint, data_folder=args.data_folder, results_folder=args.results_folder)


if __name__ == "__main__":
    main()
