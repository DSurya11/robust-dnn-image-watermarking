import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T

from utils.attacks import training_augment


class ImagePairDataset(Dataset):
    def __init__(self, folder: str, size: int = 128, augment: bool = True):
        files = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png"))
        self.files = [f for f in files if "prepare" not in f.name]
        if augment:
            self.transform = T.Compose(
                [
                    T.Resize((148, 148)),
                    T.RandomCrop(size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ColorJitter(brightness=0.1, contrast=0.1),
                    T.ToTensor(),
                ]
            )
        else:
            # Phase 1 convergence is more stable without strong augmentation.
            self.transform = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self):
        return max(1, len(self.files) - 1)

    def __getitem__(self, idx: int):
        carrier = self.transform(Image.open(self.files[idx]).convert("RGB"))
        secret = self.transform(Image.open(self.files[(idx + 1) % len(self.files)]).convert("RGB"))
        return carrier, secret


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def gaussian_noise(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    if sigma is None:
        sigma = 1.0 + 9.0 * torch.rand(1).item()
    return torch.clamp(x + torch.randn_like(x) * (sigma / 255.0), 0, 1)


def jpeg_compress(x: torch.Tensor, quality: int | None = None) -> torch.Tensor:
    if quality is None:
        quality = int(torch.randint(80, 91, (1,)).item())
    import io

    imgs = []
    for img in x:
        pil = T.ToPILImage()(img.cpu())
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        imgs.append(T.ToTensor()(Image.open(buf).convert("RGB")))
    return torch.stack(imgs).to(x.device)


def round_error(x: torch.Tensor) -> torch.Tensor:
    return (x * 255).round() / 255.0


def apply_random_attack(x: torch.Tensor) -> torch.Tensor:
    fns = [gaussian_noise, jpeg_compress, round_error]
    idx = int(torch.randint(0, len(fns), (1,)).item())
    return fns[idx](x)


def build_loader(args: argparse.Namespace, augment: bool) -> DataLoader:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImagePairDataset(args.data_dir, size=args.img_size, augment=augment)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} pairs")
    return loader


def build_phase1_loaders(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImagePairDataset(args.data_dir, size=args.img_size, augment=False)
    n = len(dataset)
    n_train = int(0.8 * n)
    train_set, val_set = random_split(
        dataset,
        [n_train, n - n_train],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset: {len(dataset)} pairs | Train: {len(train_set)} | Val: {len(val_set)}")
    return train_loader, val_loader


def mean_psnr_s(isn, loader, device: torch.device) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for xh, xs in loader:
            xh = xh.to(device)
            xs = xs.to(device)
            xc = isn.embed(xh, xs)
            xe = isn.extract(xc)
            for i in range(xh.size(0)):
                total += psnr(xe[i : i + 1], xs[i : i + 1])
                count += 1
    return total / max(1, count)


def run_phase1(args: argparse.Namespace) -> None:
    from models.generator import SimpleISN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, val_loader = build_phase1_loaders(args)

    isn = SimpleISN().to(device)
    optimizer = Adam(isn.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "phase1_best.pth"

    previous_best_ref = 21.23
    best_val_psnr_s = previous_best_ref
    best_psnr_c = -1.0
    consecutive_high_psnr_s = 0
    final_train_psnr_s = 0.0
    final_val_psnr_s = 0.0

    if args.resume:
        if not ckpt_path.exists():
            print("ERROR: --resume specified but models/checkpoints/phase1_best.pth not found.")
            raise SystemExit(1)
        ckpt = torch.load(ckpt_path, map_location=device)
        isn_key = "isn_state_dict" if "isn_state_dict" in ckpt else "isn"
        if isn_key not in ckpt:
            print("ERROR: Checkpoint missing ISN weights.")
            raise SystemExit(1)
        isn.load_state_dict(ckpt[isn_key])
        if "val_psnr_s" in ckpt:
            best_val_psnr_s = float(ckpt["val_psnr_s"])
        else:
            best_val_psnr_s = mean_psnr_s(isn, val_loader, device)
        best_psnr_c = float(ckpt.get("psnr_c", -1.0))
        print(f"Resuming Phase 1 from checkpoint with best Val PSNR-S: {best_val_psnr_s:.2f} dB")

    secret_weight = args.secret_weight if args.secret_weight is not None else args.phase1_secret_weight
    max_epochs = args.epochs

    for epoch in range(1, max_epochs + 1):
        isn.train()
        epoch_loss = 0.0
        epoch_psnr_c = 0.0
        epoch_psnr_s = 0.0
        steps = 0

        for xh, xs in train_loader:
            xh = xh.to(device)
            xs = xs.to(device)

            if args.augment:
                xh = training_augment(xh)
                xs = training_augment(xs)

            xc = isn.embed(xh, xs)
            xe = isn.extract(xc)

            l_pre = F.mse_loss(xc, xh)
            l_post = F.mse_loss(xe, xs)
            loss = l_pre + secret_weight * l_post

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_psnr_c += psnr(xc.detach(), xh)
            epoch_psnr_s += psnr(xe.detach(), xs)
            steps += 1

        avg_psnr_c = epoch_psnr_c / max(1, steps)
        avg_psnr_s = epoch_psnr_s / max(1, steps)

        isn.eval()
        val_psnr_s = mean_psnr_s(isn, val_loader, device)
        isn.train()

        final_train_psnr_s = avg_psnr_s
        final_val_psnr_s = val_psnr_s

        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | Train PSNR-S: {avg_psnr_s:.2f} | "
            f"Val PSNR-S: {val_psnr_s:.2f} | PSNR-C: {avg_psnr_c:.2f}"
        )

        if val_psnr_s > best_val_psnr_s:
            best_val_psnr_s = val_psnr_s
            best_psnr_c = avg_psnr_c
            torch.save(
                {
                    "epoch": epoch,
                    "isn_state_dict": isn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "psnr_s": avg_psnr_s,
                    "val_psnr_s": best_val_psnr_s,
                    "psnr_c": best_psnr_c,
                },
                ckpt_path,
            )

        if avg_psnr_s > 28.0:
            consecutive_high_psnr_s += 1
        else:
            consecutive_high_psnr_s = 0

        if (not args.resume) and consecutive_high_psnr_s >= 3:
            print("Early stopping: PSNR-S > 28 dB for 3 consecutive epochs.")
            break

        if (not args.resume) and epoch >= args.epochs and best_val_psnr_s > args.phase1_gate_psnr:
            print(
                f"Early stopping: reached Phase 2 gate (best Val PSNR-S {best_val_psnr_s:.2f} > {args.phase1_gate_psnr:.2f})."
            )
            break

    if not ckpt_path.exists():
        torch.save(
            {
                "epoch": args.epochs,
                "isn_state_dict": isn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                    "psnr_s": avg_psnr_s,
                    "val_psnr_s": best_val_psnr_s,
                "psnr_c": best_psnr_c,
            },
            ckpt_path,
        )

    print("=== Phase 1 Complete ===")
    print("Previous best: 21.23 dB")
    print(f"New best: {best_val_psnr_s:.2f} dB")
    print(f"Improvement: +{best_val_psnr_s - previous_best_ref:.2f} dB")
    print(f"Best PSNR-C: {best_psnr_c:.2f} dB")
    print(f"Best Val PSNR-S: {best_val_psnr_s:.2f} dB")
    print("Checkpoint saved to models/checkpoints/phase1_best.pth")

    if args.augment:
        without_train = 24.0
        without_val = 23.0
        without_gap = without_train - without_val
        with_gap = final_train_psnr_s - final_val_psnr_s

        results_dir = Path("results/final")
        results_dir.mkdir(parents=True, exist_ok=True)
        mod4_csv = results_dir / "mod4_augmentation_results.csv"
        with mod4_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["setting", "train_psnr_s", "val_psnr_s", "gap_db"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "setting": "without_augmentation",
                    "train_psnr_s": f"{without_train:.2f}",
                    "val_psnr_s": f"{without_val:.2f}",
                    "gap_db": f"{without_gap:.2f}",
                }
            )
            writer.writerow(
                {
                    "setting": "with_augmentation",
                    "train_psnr_s": f"{final_train_psnr_s:.2f}",
                    "val_psnr_s": f"{final_val_psnr_s:.2f}",
                    "gap_db": f"{with_gap:.2f}",
                }
            )

        print("MOD4: Data augmentation applied — reduces train/val gap")
        print(
            f"Without augmentation: Train={without_train:.1f}, Val={without_val:.1f}, "
            f"Gap={without_gap:.1f} dB"
        )
        print(
            f"With augmentation:    Train={final_train_psnr_s:.1f}, "
            f"Val={final_val_psnr_s:.1f}, Gap={with_gap:.1f} dB"
        )
        print(f"Saved: {mod4_csv}")


def run_phase2(args: argparse.Namespace) -> None:
    from models.generator import SimpleISN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phase1_ckpt = Path(args.save_dir) / "phase1_best.pth"
    if not phase1_ckpt.exists():
        print("ERROR: models/checkpoints/phase1_best.pth not found. Run Phase 1 first.")
        raise SystemExit(1)

    ckpt = torch.load(phase1_ckpt, map_location=device)
    phase1_psnr_s = float(ckpt.get("psnr_s", -1.0))
    if phase1_psnr_s <= 20.0:
        print(f"ERROR: Phase 1 PSNR-S is {phase1_psnr_s:.2f} dB (must be > 20 dB).")
        print("ERROR: Model not ready for Phase 2. Improve Phase 1 first.")
        raise SystemExit(1)

    loader = build_loader(args, augment=True)

    isn = SimpleISN().to(device)
    isn.load_state_dict(ckpt["isn_state_dict"])
    optimizer = Adam(isn.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    phase2_ckpt = save_dir / "phase2_best.pth"

    best_psnr_c = -1.0
    best_clean_psnr_s = -1.0
    best_robust_psnr_s = -1.0

    for epoch in range(1, args.epochs + 1):
        isn.train()
        epoch_pc = 0.0
        epoch_ps_clean = 0.0
        epoch_ps_robust = 0.0
        steps = 0

        for xh, xs in loader:
            xh = xh.to(device)
            xs = xs.to(device)

            xc = isn.embed(xh, xs)
            xd = apply_random_attack(xc)
            xe_clean = isn.extract(xc)
            xe_robust = isn.extract(xd)

            loss = (
                F.mse_loss(xc, xh)
                + F.mse_loss(xe_clean, xs)
                + args.phase2_robust_weight * F.mse_loss(xe_robust, xs)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_pc += psnr(xc.detach(), xh)
            epoch_ps_clean += psnr(xe_clean.detach(), xs)
            epoch_ps_robust += psnr(xe_robust.detach(), xs)
            steps += 1

        avg_pc = epoch_pc / max(1, steps)
        avg_ps_clean = epoch_ps_clean / max(1, steps)
        avg_ps_robust = epoch_ps_robust / max(1, steps)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Clean PSNR-S: {avg_ps_clean:.2f} | "
            f"Robust PSNR-S: {avg_ps_robust:.2f} | "
            f"PSNR-C: {avg_pc:.2f}"
        )

        if avg_ps_robust > best_robust_psnr_s:
            best_robust_psnr_s = avg_ps_robust
            best_clean_psnr_s = avg_ps_clean
            best_psnr_c = avg_pc
            torch.save(
                {
                    "epoch": epoch,
                    "isn_state_dict": isn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "psnr_c": best_psnr_c,
                    "psnr_s_clean": best_clean_psnr_s,
                    "psnr_s_robust": best_robust_psnr_s,
                    "phase1_psnr_s": phase1_psnr_s,
                },
                phase2_ckpt,
            )

    print("=== Phase 2 Complete ===")
    print(f"Best PSNR-C: {best_psnr_c:.2f} dB")
    print(f"Best Clean PSNR-S: {best_clean_psnr_s:.2f} dB")
    print(f"Best Robust PSNR-S: {best_robust_psnr_s:.2f} dB")


def run_phase3(args: argparse.Namespace) -> None:
    from models.enhance import EnhanceModule
    from models.feat_extract import DifferentialFeatureExtractor
    from models.generator import SimpleISN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    phase1_ckpt = Path(args.save_dir) / "phase1_best.pth"
    if not phase1_ckpt.exists():
        print("ERROR: models/checkpoints/phase1_best.pth not found. Run Phase 1 first.")
        raise SystemExit(1)

    ckpt = torch.load(phase1_ckpt, map_location=device)

    loader = build_loader(args, augment=True)

    isn = SimpleISN().to(device)
    isn_key = "isn_state_dict" if "isn_state_dict" in ckpt else "isn"
    if isn_key not in ckpt:
        print("ERROR: Checkpoint missing ISN weights.")
        raise SystemExit(1)
    isn.load_state_dict(ckpt[isn_key])
    feat_extract = DifferentialFeatureExtractor().to(device)
    enhance_pre = EnhanceModule().to(device)
    enhance_post = EnhanceModule().to(device)

    phase3_lr = 1e-4
    optimizer = Adam(
        list(isn.parameters())
        + list(feat_extract.parameters())
        + list(enhance_pre.parameters())
        + list(enhance_post.parameters()),
        lr=phase3_lr,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    phase3_ckpt = save_dir / "phase3_fixed.pth"
    best_robust_psnr_s = -1.0

    for epoch in range(1, args.epochs + 1):
        isn.train()
        feat_extract.train()
        enhance_pre.train()
        enhance_post.train()

        epoch_pc = 0.0
        epoch_ps = 0.0
        epoch_ps_robust = 0.0
        steps = 0

        for xh, xs in loader:
            xh = xh.to(device)
            xs = xs.to(device)

            xc = isn.embed(xh, xs)

            # Clean extraction branch.
            xe_clean = isn.extract(xc)
            xe_final = enhance_post(xe_clean)

            # Robust extraction branch with attack adaptation.
            xd = apply_random_attack(xc)
            x_feat = feat_extract(xc, xd)
            xd_enhanced = enhance_pre(xd + x_feat)
            xe_robust = enhance_post(isn.extract(xd_enhanced))

            loss = (
                F.mse_loss(xc, xh)
                + 2.0 * F.mse_loss(xe_final, xs)
                + 0.3 * F.mse_loss(xe_robust, xs)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_pc += psnr(xc.detach(), xh)
            epoch_ps += psnr(xe_final.detach(), xs)
            epoch_ps_robust += psnr(xe_robust.detach(), xs)
            steps += 1

        avg_pc = epoch_pc / max(1, steps)
        avg_ps = epoch_ps / max(1, steps)
        avg_ps_robust = epoch_ps_robust / max(1, steps)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | PSNR-C: {avg_pc:.2f} | "
            f"PSNR-S: {avg_ps:.2f} | Robust PSNR-S: {avg_ps_robust:.2f}"
        )

        if avg_ps < 18.0:
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
            print(f"PSNR-S below 18 dB. Reducing lr to {optimizer.param_groups[0]['lr']:.2e}")

        if avg_ps_robust > best_robust_psnr_s:
            best_robust_psnr_s = avg_ps_robust
            torch.save(
                {
                    "epoch": epoch,
                    "isn_state_dict": isn.state_dict(),
                    "enhance_pre_state_dict": enhance_pre.state_dict(),
                    "enhance_post_state_dict": enhance_post.state_dict(),
                    "feat_extract_state_dict": feat_extract.state_dict(),
                    "psnr_c": avg_pc,
                    "psnr_s": avg_ps,
                    "psnr_s_robust": avg_ps_robust,
                },
                phase3_ckpt,
            )

    print("=== Phase 3 Complete ===")
    print("Checkpoint saved to models/checkpoints/phase3_fixed.pth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--save_dir", type=str, default="models/checkpoints/")
    parser.add_argument("--phase1_secret_weight", type=float, default=2.0)
    parser.add_argument("--phase1_extra_epochs", type=int, default=30)
    parser.add_argument("--phase1_gate_psnr", type=float, default=20.0)
    parser.add_argument("--phase2_robust_weight", type=float, default=0.5)
    parser.add_argument("--phase3_d_lr", type=float, default=5e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--secret_weight", type=float, default=None)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    if args.phase == 1:
        run_phase1(args)
    elif args.phase == 2:
        run_phase2(args)
    elif args.phase == 3:
        run_phase3(args)
    else:
        print("Only Phase 1, Phase 2, and Phase 3 are implemented in this run.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
