"""Visualization utilities for watermarking experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
