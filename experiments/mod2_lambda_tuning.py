"""MOD 2: Lambda tradeoff via fast fine-tuning from trained SimpleISN."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RESULTS_DIR = Path("results/final")
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
            xe = isn.extract(xc)
            psnr_c_vals.append(psnr(xc, xh))
            psnr_s_vals.append(psnr(xe, xs))
    return {
        "PSNR-C": float(sum(psnr_c_vals) / max(1, len(psnr_c_vals))),
        "PSNR-S": float(sum(psnr_s_vals) / max(1, len(psnr_s_vals))),
    }


def build_train_val_loaders(batch_size: int = 4, num_pairs: int = 50):
    ds = PairDataset(num_pairs=num_pairs, size=128)
    n = len(ds)
    n_train = int(0.8 * n)
    train_set, val_set = random_split(
        ds,
        [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_epochs", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/checkpoints/phase1_best.pth"
    if not Path(model_path).exists():
        print("ERROR: models/checkpoints/phase1_best.pth not found")
        sys.exit(1)

    clean_psnr = get_clean_psnr(model_path)
    print("=== Running Modification 2 ===")
    print(f"Model checkpoint: {Path(model_path).name}")
    print(f"Sanity check PSNR-S: {clean_psnr:.2f} dB")
    print("Proceeding...")

    train_loader, val_loader = build_train_val_loaders(batch_size=4, num_pairs=50)

    rows: List[Dict[str, float]] = []
    for lambda_c, lambda_s in CONFIGS:
        # Fresh model copy from phase1_best for each lambda pair.
        isn = load_isn(device, model_path)
        isn.train()
        opt = Adam(isn.parameters(), lr=1e-4)

        for _ in range(args.finetune_epochs):
            for xh, xs in train_loader:
                xh = xh.to(device)
                xs = xs.to(device)
                xc = isn.embed(xh, xs)
                xe = isn.extract(xc)

                l_pre = torch.nn.functional.mse_loss(xc, xh)
                l_post = torch.nn.functional.mse_loss(xe, xs)
                loss = lambda_c * l_pre + lambda_s * l_post

                opt.zero_grad()
                loss.backward()
                opt.step()

        metrics = evaluate_config(isn, val_loader, device)
        print(
            f"Lambda (lc={lambda_c:.1f}, ls={lambda_s:.1f}): "
            f"PSNR-C={metrics['PSNR-C']:.2f}, PSNR-S={metrics['PSNR-S']:.2f}"
        )
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
    csv_path = RESULTS_DIR / "mod2_lambda_results.csv"
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
    ax1.text(1.02, max(y_c), "paper default", color="gray", fontsize=9)
    ax1.set_xlabel("lambda_c")
    ax1.set_ylabel("PSNR-C (dB)", color="tab:blue")
    ax2.set_ylabel("PSNR-S (dB)", color="tab:orange")
    ax1.set_title("MOD 2: Lambda Tradeoff Curve")
    fig.tight_layout()
    chart_path = RESULTS_DIR / "mod2_lambda_chart.png"
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
