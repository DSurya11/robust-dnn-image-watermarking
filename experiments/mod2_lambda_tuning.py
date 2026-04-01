"""MOD 2: Lambda tradeoff via fast fine-tuning from trained SimpleISN."""

import argparse
import csv
import copy
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from run_forward import apply_attack


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
CONFIGS: List[Tuple[float, float]] = [
    (0.3, 1.7),
    (0.5, 1.5),
    (0.7, 1.3),
    (1.0, 1.0),
    (1.3, 0.7),
    (1.5, 0.5),
    (1.7, 0.3),
]


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
    return isn


class PairDataset(Dataset):
    def __init__(self, num_pairs: int = 8, size: int = 128):
        files = sorted(DATA_DIR.glob("*.jpg")) + sorted(DATA_DIR.glob("*.png"))
        files = [f for f in files if "prepare" not in f.name]
        self.tfm = T.Compose([T.Resize((size, size)), T.ToTensor()])
        self.pairs = []
        max_pairs = min(num_pairs, max(1, len(files) - 1))
        for i in range(max_pairs):
            c = self.tfm(Image.open(files[i]).convert("RGB"))
            s = self.tfm(Image.open(files[(i + 1) % len(files)]).convert("RGB"))
            self.pairs.append((c, s))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def get_clean_psnr(model_path: str) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isn = load_isn(device, model_path)
    isn.eval()
    ds = PairDataset(num_pairs=1, size=128)
    if len(ds) == 0:
        return 0.0
    with torch.no_grad():
        xh, xs = ds[0]
        xh = xh.unsqueeze(0).to(device)
        xs = xs.unsqueeze(0).to(device)
        xc = isn.embed(xh, xs)
        xe = isn.extract(xc)
        return psnr(xe, xs)


def evaluate_config(isn, loader, device: torch.device) -> Dict[str, float]:
    isn.eval()
    psnr_c_vals: List[float] = []
    psnr_s_vals: List[float] = []
    with torch.no_grad():
        for xh, xs in loader:
            xh = xh.to(device)
            xs = xs.to(device)
            xc = isn.embed(xh, xs)
            xd = apply_attack(xc, "Gaussian_s10")
            xe = isn.extract(xd)
            psnr_c_vals.append(psnr(xc, xh))
            psnr_s_vals.append(psnr(xe, xs))
    return {
        "PSNR-C": float(sum(psnr_c_vals) / max(1, len(psnr_c_vals))),
        "PSNR-S": float(sum(psnr_s_vals) / max(1, len(psnr_s_vals))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/phase3_final.pth")
    parser.add_argument("--finetune_epochs", type=int, default=10)
    args = parser.parse_args()

    model_path = resolve_checkpoint(args.checkpoint)
    if model_path is None:
        print("ERROR: Model not trained properly. Run train.py first.")
        sys.exit(1)

    clean_psnr = get_clean_psnr(model_path)
    print("=== Running Modification 2 ===")
    print(f"Model checkpoint: {Path(model_path).name}")
    print(f"Sanity check PSNR-S: {clean_psnr:.2f} dB")
    print("Proceeding...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_isn = load_isn(device, model_path)
    base_state = copy.deepcopy(base_isn.state_dict())

    ds = PairDataset(num_pairs=8, size=128)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    rows: List[Dict[str, float]] = []
    for lambda_c, lambda_s in CONFIGS:
        isn = load_isn(device, model_path)
        isn.load_state_dict(base_state)
        isn.train()
        opt = Adam(isn.parameters(), lr=2e-4)

        # Fast adaptation from trained weights, not training from scratch.
        for _ in range(args.finetune_epochs):
            for xh, xs in loader:
                xh = xh.to(device)
                xs = xs.to(device)
                xc = isn.embed(xh, xs)
                xe = isn.extract(xc)
                xd = apply_attack(xc.detach(), "Gaussian_s10")
                xe_robust = isn.extract(xd)

                l_pre = torch.nn.functional.mse_loss(xc, xh)
                l_post = torch.nn.functional.mse_loss(xe, xs)
                l_robust = torch.nn.functional.mse_loss(xe_robust, xs)
                loss = lambda_c * l_pre + lambda_s * l_post + 0.5 * l_robust

                opt.zero_grad()
                loss.backward()
                opt.step()

        metrics = evaluate_config(isn, loader, device)
        rows.append(
            {
                "lambda_c": lambda_c,
                "lambda_s": lambda_s,
                "PSNR-C": metrics["PSNR-C"],
                "PSNR-S": metrics["PSNR-S"],
                "Combined": metrics["PSNR-C"] + metrics["PSNR-S"],
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "mod2_lambda_results_v2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda_c", "lambda_s", "PSNR-C", "PSNR-S", "Combined"])
        writer.writeheader()
        writer.writerows(rows)

    x_vals = [r["lambda_c"] for r in rows]
    y_c = [r["PSNR-C"] for r in rows]
    y_s = [r["PSNR-S"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(x_vals, y_c, marker="o", color="tab:blue", label="PSNR-C")
    ax2.plot(x_vals, y_s, marker="s", color="tab:orange", label="PSNR-S")
    ax1.axvline(1.0, linestyle="--", color="gray", linewidth=1.3)
    ax1.set_xlabel("lambda_c")
    ax1.set_ylabel("PSNR-C (dB)", color="tab:blue")
    ax2.set_ylabel("PSNR-S (dB)", color="tab:orange")
    ax1.set_title("MOD 2: Lambda Tradeoff Curve (v2)")
    fig.tight_layout()
    chart_path = RESULTS_DIR / "mod2_lambda_chart_v2.png"
    plt.savefig(chart_path, dpi=200)
    plt.close(fig)

    best = max(rows, key=lambda r: r["Combined"])
    baseline = next(r for r in rows if abs(r["lambda_c"] - 1.0) < 1e-9 and abs(r["lambda_s"] - 1.0) < 1e-9)
    improvement = best["Combined"] - baseline["Combined"]
    print(
        f"Optimal config: lambda_c={best['lambda_c']:.1f}, lambda_s={best['lambda_s']:.1f} | "
        f"combined improvement over default: {improvement:.2f} dB"
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {chart_path}")


if __name__ == "__main__":
    main()
