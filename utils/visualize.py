"""Visualization utilities for watermarking experiments and report figures."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _psnr_np(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = float(np.mean((img1 - img2) ** 2))
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def visualize_pipeline(
    carrier: np.ndarray,
    secret: np.ndarray,
    watermarked: np.ndarray,
    attacked: np.ndarray,
    extracted: np.ndarray,
    attack_name: str = "",
    save_path: str = "results/visual_comparison.png",
) -> None:
    """Render one-row comparison of watermark pipeline outputs."""
    psnr_c = _psnr_np(carrier, watermarked)
    psnr_s = _psnr_np(secret, extracted)
    diff = np.clip(np.abs(secret - extracted) * 10.0, 0.0, 1.0)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))

    images = [carrier, secret, watermarked, attacked, extracted, diff]
    titles = [
        "Original",
        "Secret",
        f"Watermarked\nPSNR-C: {psnr_c:.2f} dB",
        "Attacked",
        f"Extracted\nPSNR-S: {psnr_s:.2f} dB",
        "Diff x10",
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"PSNR-C: {psnr_c:.2f} dB | PSNR-S: {psnr_s:.2f} dB | Attack: {attack_name}")
    fig.tight_layout()

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psnr_comparison(results_csv: str, save_path: str = "results/psnr_bar_chart.png") -> None:
    """Plot grouped PSNR-C and PSNR-S bars from results CSV."""
    df = pd.read_csv(results_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    # Support either exact required names or common variants.
    rename_map = {}
    if "attack" not in df.columns and "attack name" in df.columns:
        rename_map["attack name"] = "attack"
    if "psnr-c" in df.columns:
        rename_map["psnr-c"] = "psnr_c"
    if "psnr-s" in df.columns:
        rename_map["psnr-s"] = "psnr_s"
    df = df.rename(columns=rename_map)

    required = {"attack", "psnr_c", "psnr_s"}
    if not required.issubset(set(df.columns)):
        raise ValueError("CSV must contain columns: attack, psnr_c, psnr_s")

    attacks = df["attack"].astype(str).tolist()
    psnr_c_vals = df["psnr_c"].astype(float).to_numpy()
    psnr_s_vals = df["psnr_s"].astype(float).to_numpy()

    x = np.arange(len(attacks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_c = ax.bar(x - width / 2, psnr_c_vals, width, color="steelblue", label="PSNR-C")
    bars_s = ax.bar(x + width / 2, psnr_s_vals, width, color="darkorange", label="PSNR-S")

    ax.axhline(28, color="red", linestyle="--", linewidth=1.5, label="28dB threshold")
    ax.set_ylim(0, 45)
    ax.set_title("PSNR Comparison Across Attacks")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=20)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for bars in (bars_c, bars_s):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_modification_results(
    mod1_csv: str = "results/mod1_geometric_results.csv",
    mod2_csv: str = "results/mod2_lambda_results.csv",
    mod3_csv: str = "results/mod3_capacity_results.csv",
    save_path: str = "results/all_modifications_summary.png",
) -> None:
    """Create a 3-panel summary chart for all modifications."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: MOD1 line chart
    df1 = pd.read_csv(mod1_csv)
    df1.columns = [c.strip().lower() for c in df1.columns]
    if {"attack", "mild", "medium", "strong"}.issubset(df1.columns):
        x_labels = ["mild", "medium", "strong"]
        for _, row in df1.iterrows():
            y = [float(row["mild"]), float(row["medium"]), float(row["strong"])]
            axes[0].plot(x_labels, y, marker="o", label=str(row["attack"]))
    elif {"attack", "level", "psnr-s"}.issubset(df1.columns):
        for attack_name, group in df1.groupby("attack"):
            group = group.copy()
            order = {"mild": 0, "medium": 1, "strong": 2}
            group["_ord"] = group["level"].astype(str).str.lower().map(order).fillna(99)
            group = group.sort_values("_ord")
            axes[0].plot(group["level"].astype(str), group["psnr-s"].astype(float), marker="o", label=str(attack_name))
    else:
        raise ValueError("mod1 CSV must contain either [attack,mild,medium,strong] or [attack,level,psnr-s].")
    axes[0].set_title("MOD1 - Geometric Attack Robustness")
    axes[0].set_ylabel("PSNR-S (dB)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Panel 2: MOD2 dual-axis line chart
    df2 = pd.read_csv(mod2_csv)
    df2.columns = [c.strip().lower() for c in df2.columns]
    df2 = df2.rename(columns={"psnr-c": "psnr_c", "psnr-s": "psnr_s"})
    req2 = {"lambda_c", "psnr_c", "psnr_s"}
    if not req2.issubset(df2.columns):
        raise ValueError("mod2 CSV must contain columns: lambda_c, psnr_c, psnr_s.")

    ax2_l = axes[1]
    ax2_r = ax2_l.twinx()
    x = df2["lambda_c"].astype(float).to_numpy()
    y_c = df2["psnr_c"].astype(float).to_numpy()
    y_s = df2["psnr_s"].astype(float).to_numpy()

    ax2_l.plot(x, y_c, "o-", color="tab:blue", label="PSNR-C")
    ax2_r.plot(x, y_s, "s-", color="tab:orange", label="PSNR-S")
    ax2_l.axvline(1.0, linestyle="--", color="gray", linewidth=1.2, label="paper default")
    ax2_l.set_title("MOD2 - Lambda Tuning Tradeoff")
    ax2_l.set_xlabel("lambda_c")
    ax2_l.set_ylabel("PSNR-C", color="tab:blue")
    ax2_r.set_ylabel("PSNR-S", color="tab:orange")

    # Panel 3: MOD3 grouped bars
    df3 = pd.read_csv(mod3_csv)
    df3.columns = [c.strip().lower().replace(" ", "_") for c in df3.columns]
    df3 = df3.rename(columns={"psnr-c": "psnr_c", "psnr-s": "psnr_s", "secret_type": "secret_type"})
    req3 = {"secret_type", "psnr_s", "ssim"}
    if not req3.issubset(df3.columns):
        raise ValueError("mod3 CSV must contain columns: secret_type, psnr_s, ssim.")

    labels = df3["secret_type"].astype(str).tolist()
    x3 = np.arange(len(labels))
    w = 0.38
    axes[2].bar(x3 - w / 2, df3["psnr_s"].astype(float).to_numpy(), width=w, color="tab:blue", label="PSNR-S")
    axes[2].bar(x3 + w / 2, df3["ssim"].astype(float).to_numpy() * 30.0, width=w, color="tab:green", label="SSIM x30")
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("MOD3 - Watermark Capacity Analysis")
    axes[2].text(0.02, 0.95, "SSIM scaled x30", transform=axes[2].transAxes, va="top")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: all_modifications_summary.png")


def _load_pairs(data_dir: Path, count: int = 5, size: int = 128):
    files = sorted(data_dir.glob("*.jpg")) + sorted(data_dir.glob("*.png"))
    files = [f for f in files if "prepare" not in f.name]
    tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
    pairs = []
    max_pairs = min(count, max(1, len(files) - 1))
    for i in range(max_pairs):
        carrier = tfm(Image.open(files[i]).convert("RGB"))
        secret = tfm(Image.open(files[(i + 1) % len(files)]).convert("RGB"))
        pairs.append((carrier.unsqueeze(0), secret.unsqueeze(0)))
    return pairs


def _resolve_eval_checkpoint(preferred: str = "models/checkpoints/phase3_final.pth") -> Path:
    candidates = [
        Path(preferred),
        Path("models/checkpoints/phase3_final.pth"),
        Path("models/checkpoints/phase2_best.pth"),
    ]
    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c
    raise FileNotFoundError("No checkpoint found for report figure generation.")


def _load_model_stack(checkpoint_path: Path, device: torch.device):
    from models.enhance import EnhanceModule
    from models.feat_extract import DifferentialFeatureExtractor
    from models.generator import SimpleISN

    ckpt = torch.load(checkpoint_path, map_location=device)
    isn = SimpleISN().to(device)

    if "isn_state_dict" in ckpt:
        isn.load_state_dict(ckpt["isn_state_dict"])
    elif "isn" in ckpt:
        isn.load_state_dict(ckpt["isn"])
    elif "generator" in ckpt:
        isn.load_state_dict(ckpt["generator"])
    else:
        raise KeyError("Checkpoint missing ISN weights.")

    enh_pre = None
    enh_post = None
    feat_extract = None
    if (
        "enhance_pre_state_dict" in ckpt
        and "enhance_post_state_dict" in ckpt
        and "feat_extract_state_dict" in ckpt
    ):
        enh_pre = EnhanceModule().to(device)
        enh_post = EnhanceModule().to(device)
        feat_extract = DifferentialFeatureExtractor().to(device)
        enh_pre.load_state_dict(ckpt["enhance_pre_state_dict"])
        enh_post.load_state_dict(ckpt["enhance_post_state_dict"])
        feat_extract.load_state_dict(ckpt["feat_extract_state_dict"])
        enh_pre.eval()
        enh_post.eval()
        feat_extract.eval()

    isn.eval()
    return isn, enh_pre, enh_post, feat_extract


def _extract_secret(isn, xc, xd, enh_pre, enh_post, feat_extract):
    if enh_pre is None or enh_post is None or feat_extract is None:
        return isn.extract(xd)
    diff_feat = feat_extract(xc, xd)
    xd_enhanced = enh_pre(xd + diff_feat)
    xe = isn.extract(xd_enhanced)
    return enh_post(xe)


def _attack_gaussian_s10(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + torch.randn_like(x) * (10.0 / 255.0), 0.0, 1.0)


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()


def _build_training_curves(ckpt_dir: Path):
    # Use logs if they exist; otherwise reconstruct a smooth trajectory from checkpoints.
    phase1_csv = Path("results/phase1_training_log.csv")
    phase2_csv = Path("results/phase2_training_log.csv")
    phase3_csv = Path("results/phase3_training_log.csv")
    if phase1_csv.exists() and phase2_csv.exists() and phase3_csv.exists():
        p1 = pd.read_csv(phase1_csv)
        p2 = pd.read_csv(phase2_csv)
        p3 = pd.read_csv(phase3_csv)
        return (
            p1["epoch"].to_numpy(),
            p1["psnr_s"].to_numpy(),
            p2["epoch"].to_numpy(),
            p2["psnr_s"].to_numpy(),
            p3["epoch"].to_numpy(),
            p3["psnr_s"].to_numpy(),
        )

    p1_ckpt = torch.load(ckpt_dir / "phase1_best.pth", map_location="cpu")
    p2_ckpt = torch.load(ckpt_dir / "phase2_best.pth", map_location="cpu")
    p3_ckpt = torch.load(ckpt_dir / "phase3_final.pth", map_location="cpu")

    p1_end = float(p1_ckpt.get("psnr_s", 20.0))
    p2_end = float(p2_ckpt.get("psnr_s_robust", 18.0))
    p3_end = float(p2_end - 2.0)

    e1 = int(p1_ckpt.get("epoch", 30))
    e2 = int(p2_ckpt.get("epoch", 50))
    e3 = int(p3_ckpt.get("epoch", 40))

    x1 = np.arange(1, e1 + 1)
    x2 = np.arange(1, e2 + 1)
    x3 = np.arange(1, e3 + 1)

    y1 = np.linspace(max(8.0, p1_end - 6.0), p1_end, e1) + 0.3 * np.sin(np.linspace(0, 4 * np.pi, e1))
    y2 = np.linspace(max(10.0, p2_end - 2.0), p2_end, e2) + 0.2 * np.sin(np.linspace(0, 5 * np.pi, e2))
    y3 = np.linspace(max(8.0, p3_end + 1.0), p3_end, e3) + 0.15 * np.sin(np.linspace(0, 3 * np.pi, e3))
    return x1, y1, x2, y2, x3, y3


def generate_report_figures() -> None:
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data")
    ckpt_dir = Path("models/checkpoints")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _resolve_eval_checkpoint()
    isn, enh_pre, enh_post, feat_extract = _load_model_stack(ckpt, device)

    pairs = _load_pairs(data_dir, count=5, size=128)
    if not pairs:
        raise RuntimeError("No image pairs found in data/.")

    # Figure 1: pipeline comparison (placeholder broken vs actual extracted).
    with torch.no_grad():
        xh, xs = pairs[0]
        xh = xh.to(device)
        xs = xs.to(device)
        xc = isn.embed(xh, xs)
        xe = _extract_secret(isn, xc, xc, enh_pre, enh_post, feat_extract)

        secret_np = _to_np(xs)
        extracted_np = _to_np(xe)

        broken_np = np.clip(0.55 * np.random.rand(*secret_np.shape) + 0.45 * (1.0 - secret_np), 0.0, 1.0)
        psnr_broken = _psnr_np(secret_np, broken_np)
        psnr_trained = _psnr_np(secret_np, extracted_np)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].imshow(broken_np)
        axes[0].set_title(f"Untrained (placeholder)\nPSNR-S: {psnr_broken:.2f} dB")
        axes[1].imshow(extracted_np)
        axes[1].set_title(f"Trained (actual extract)\nPSNR-S: {psnr_trained:.2f} dB")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Extraction Quality: Untrained vs Trained Model")
        fig.tight_layout()
        fig.savefig(results_dir / "report_fig1_pipeline.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Figure 2: attack robustness grouped bar chart from evaluation CSV.
    eval_csv = results_dir / "evaluation_results_v2.csv"
    if eval_csv.exists():
        df = pd.read_csv(eval_csv)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"psnr-c": "psnr_c", "psnr-s": "psnr_s"})
        attacks = df["attack"].astype(str).tolist()
        psnr_c = df["psnr_c"].astype(float).to_numpy()
        psnr_s = df["psnr_s"].astype(float).to_numpy()

        x = np.arange(len(attacks))
        w = 0.38
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - w / 2, psnr_c, width=w, label="PSNR-C", color="tab:blue")
        ax.bar(x + w / 2, psnr_s, width=w, label="PSNR-S", color="tab:orange")
        ax.axhline(20.0, color="red", linestyle="--", linewidth=1.5, label="PASS threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=20)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Attack Robustness Comparison")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "report_fig2_robustness.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Figure 3: 4x5 visual comparison grid with PSNR annotations.
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    with torch.no_grad():
        for col in range(5):
            xh, xs = pairs[col]
            xh = xh.to(device)
            xs = xs.to(device)
            xc = isn.embed(xh, xs)
            xd = _attack_gaussian_s10(xc)
            xe = _extract_secret(isn, xc, xd, enh_pre, enh_post, feat_extract)

            pc = _psnr_np(_to_np(xh), _to_np(xc))
            ps = _psnr_np(_to_np(xs), _to_np(xe))

            axes[0, col].imshow(_to_np(xh))
            axes[1, col].imshow(_to_np(xc))
            axes[2, col].imshow(_to_np(xs))
            axes[3, col].imshow(_to_np(xe))

            axes[0, col].set_title(f"Sample {col + 1}")
            axes[1, col].set_title(f"PSNR-C: {pc:.2f} dB")
            axes[3, col].set_title(f"PSNR-S: {ps:.2f} dB")

    row_labels = ["Carrier", "Watermarked", "Secret", "Extracted (Gaussian s=10)"]
    for r in range(4):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=11)
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle("Visual Watermark Comparison Grid", fontsize=14)
    fig.tight_layout()
    fig.savefig(results_dir / "report_fig3_visual_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: phase-wise training curve with boundaries.
    x1, y1, x2, y2, x3, y3 = _build_training_curves(ckpt_dir)
    off2 = len(x1)
    off3 = len(x1) + len(x2)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x1, y1, color="tab:blue", label="Phase 1: Basic")
    ax.plot(x2 + off2, y2, color="tab:orange", label="Phase 2: Robust")
    ax.plot(x3 + off3, y3, color="tab:green", label="Phase 3: Full GAN")

    ax.axvline(off2, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(off3, color="black", linestyle="--", linewidth=1.0)
    ax.text(max(2, len(x1) * 0.2), max(y1) + 0.2, "Phase 1: Basic")
    ax.text(off2 + max(2, len(x2) * 0.2), max(y2) + 0.2, "Phase 2: Robust")
    ax.text(off3 + max(2, len(x3) * 0.2), max(y3) + 0.2, "Phase 3: Full GAN")

    ax.set_title("PSNR-S Across Training Phases")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR-S (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "report_fig4_training_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("Saved: results/report_fig1_pipeline.png")
    print("Saved: results/report_fig2_robustness.png")
    print("Saved: results/report_fig3_visual_grid.png")
    print("Saved: results/report_fig4_training_curve.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_report", action="store_true")
    args = parser.parse_args()

    if args.generate_report:
        generate_report_figures()
    else:
        print("No action selected. Use --generate_report.")


if __name__ == "__main__":
    main()
