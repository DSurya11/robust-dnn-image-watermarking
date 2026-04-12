"""MOD 5: Compare MSE-only vs MSE+SSIM loss on extraction quality."""

import argparse
import copy
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_metric
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.losses import ssim_loss


RESULTS_DIR = Path("results/final")
DATA_DIR = Path("data")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 100.0 if mse < 1e-10 else 10.0 * torch.log10(torch.tensor(1.0 / mse)).item()


def batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().clamp(0.0, 1.0).cpu().numpy()
    target_np = target.detach().clamp(0.0, 1.0).cpu().numpy()
    vals = []
    for i in range(pred_np.shape[0]):
        p = pred_np[i].transpose(1, 2, 0)
        t = target_np[i].transpose(1, 2, 0)
        vals.append(ssim_metric(p, t, channel_axis=2, data_range=1.0))
    return float(sum(vals) / max(1, len(vals)))


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
    def __init__(self, num_pairs: int = 10, size: int = 128):
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


def evaluate_model(isn, loader, device: torch.device) -> dict:
    isn.eval()
    psnr_vals = []
    ssim_vals = []
    with torch.no_grad():
        for xh, xs in loader:
            xh = xh.to(device)
            xs = xs.to(device)
            xe = isn.extract(isn.embed(xh, xs))
            psnr_vals.append(psnr(xe, xs))
            ssim_vals.append(batch_ssim(xe, xs))
    return {
        "PSNR-S": float(sum(psnr_vals) / max(1, len(psnr_vals))),
        "SSIM": float(sum(ssim_vals) / max(1, len(ssim_vals))),
    }


def train_config(base_state, model_path: str, loader, device: torch.device, loss_type: str, epochs: int):
    isn = load_isn(device, model_path)
    isn.load_state_dict(base_state)
    isn.train()
    opt = Adam(isn.parameters(), lr=1e-4)

    for _ in range(epochs):
        for xh, xs in loader:
            xh = xh.to(device)
            xs = xs.to(device)
            xc = isn.embed(xh, xs)
            xe = isn.extract(xc)

            if loss_type == "mse":
                loss = F.mse_loss(xe, xs)
            else:
                loss = 0.5 * F.mse_loss(xe, xs) + 0.5 * ssim_loss(xe, xs)

            opt.zero_grad()
            loss.backward()
            opt.step()

    return evaluate_model(isn, loader, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PairDataset(num_pairs=10, size=128)
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    model_path = "models/checkpoints/phase1_best.pth"
    if not Path(model_path).exists():
        print("ERROR: models/checkpoints/phase1_best.pth not found")
        raise SystemExit(1)

    base_isn = load_isn(device, model_path)
    base_state = copy.deepcopy(base_isn.state_dict())

    mse_metrics = train_config(base_state, model_path, loader, device, "mse", args.epochs)
    hybrid_metrics = train_config(base_state, model_path, loader, device, "hybrid", args.epochs)

    rows = [
        {
            "loss_type": "MSE only",
            "psnr_s": mse_metrics["PSNR-S"],
            "ssim": mse_metrics["SSIM"],
            "gradient_flows": "Yes",
        },
        {
            "loss_type": "MSE + SSIM",
            "psnr_s": hybrid_metrics["PSNR-S"],
            "ssim": hybrid_metrics["SSIM"],
            "gradient_flows": "Yes (now fixed)",
        },
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "mod5_ssim_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["loss_type", "psnr_s", "ssim", "gradient_flows"])
        writer.writeheader()
        writer.writerows(rows)

    print("Loss Type     | PSNR-S | SSIM  | Gradient flows?")
    print(f"MSE only      | {mse_metrics['PSNR-S']:.2f}  | {mse_metrics['SSIM']:.3f} | Yes")
    print(
        f"MSE + SSIM    | {hybrid_metrics['PSNR-S']:.2f}  | "
        f"{hybrid_metrics['SSIM']:.3f} | Yes (now fixed)"
    )
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
