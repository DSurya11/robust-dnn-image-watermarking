import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import autocast, GradScaler
import random
import time
import os
import math
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class ImagePairDataset(Dataset):
    def __init__(self, folder, size=128):
        self.files = sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png"))
        self.files = [f for f in self.files if "prepare" not in f.name]
        self.transform = T.Compose([
            T.Resize((148, 148)),
            T.RandomCrop(128),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
        ])

    def __len__(self):
        return max(1, len(self.files) - 1)

    def __getitem__(self, i):
        carrier = self.transform(Image.open(self.files[i]).convert("RGB"))
        secret = self.transform(Image.open(self.files[(i + 1) % len(self.files)]).convert("RGB"))
        return carrier, secret


def psnr(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def ssim_loss(img1, img2):
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    s1 = F.avg_pool2d(img1**2, 3, 1, 1) - mu1_sq
    s2 = F.avg_pool2d(img2**2, 3, 1, 1) - mu2_sq
    s12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu1_mu2 + c1) * (2 * s12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (s1 + s2 + c2)
    return (num / den).mean()


def gaussian_noise(x, sigma=None):
    if sigma is None:
        sigma = random.uniform(1, 10)
    return torch.clamp(x + torch.randn_like(x) * (sigma / 255.0), 0, 1)


def jpeg_compress(x, quality=None):
    if quality is None:
        quality = random.randint(80, 90)
    import io

    imgs = []
    for img in x:
        pil = T.ToPILImage()(img.cpu())
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=quality)
        buf.seek(0)
        imgs.append(T.ToTensor()(Image.open(buf)))
    return torch.stack(imgs).to(x.device)


def round_error(x):
    return (x * 255).round() / 255.0


def random_attack(x):
    fn = random.choice([gaussian_noise, jpeg_compress, round_error])
    return fn(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    torch.backends.cudnn.benchmark = True

    from models.embedder import WatermarkEmbedder
    from models.extractor import WatermarkExtractor
    from models.discriminator import Discriminator
    from models.feat_extract import DifferentialFeatureExtractor
    from models.enhance import EnhanceModule

    embedder = WatermarkEmbedder().to(device)
    extractor = WatermarkExtractor().to(device)
    feat_ext = DifferentialFeatureExtractor().to(device)
    enh_pre = EnhanceModule(window_size=4).to(device)
    enh_post = EnhanceModule(window_size=8).to(device)
    _disc = Discriminator().to(device)

    dataset = ImagePairDataset("data/", size=128)
    BATCH_SIZE = 4
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    print(f"Dataset: {len(dataset)} pairs")

    scaler = GradScaler("cuda")
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for f in ckpt_dir.glob("*.pth"):
        f.unlink()
        print(f"Deleted old checkpoint: {f.name}")
    print("Starting fresh training...")

    best_psnr_s = 0.0
    avg_psnr_s = 0.0
    start_time = time.time()

    NUM_EPOCHS = 200

    params_main = list(embedder.parameters()) + list(extractor.parameters()) + list(feat_ext.parameters()) + list(enh_pre.parameters()) + list(enh_post.parameters())
    opt = Adam(params_main, lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)

    for epoch in range(1, NUM_EPOCHS + 1):
        if avg_psnr_s > 28.0:
            print("Early stop: avg_psnr_s exceeded 28.0")
            break

        phase = 1

        epoch_loss = 0.0
        epoch_psnr_c = 0.0
        epoch_psnr_s = 0.0
        steps = 0

        for step, (carrier, secret) in enumerate(loader):
            carrier = carrier.to(device)
            secret = secret.to(device)

            with autocast("cuda", enabled=device.type == "cuda"):
                watermarked = embedder(carrier, secret)
                attacked = random_attack(watermarked.detach())
                diff_feat = feat_ext(watermarked, attacked)
                enhanced = enh_pre(attacked)
                enhanced = enh_post(enhanced)
                extracted = extractor(enhanced, diff_feat)

                loss_c = F.mse_loss(watermarked, carrier)
                loss_s = F.mse_loss(extracted, secret)
                loss_f = F.mse_loss(watermarked, attacked)
                loss_ssim = ssim_loss(watermarked, carrier)

                loss_total = 1.0 * loss_c + 1.5 * loss_s + 0.05 * loss_f + 0.1 * (1 - loss_ssim)

            opt.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.step(opt)
            scaler.update()

            pc = psnr(watermarked.detach(), carrier)
            ps = psnr(extracted.detach(), secret)
            epoch_psnr_c += pc
            epoch_psnr_s += ps
            epoch_loss += loss_total.item()
            steps += 1

            if step % 10 == 0:
                mem = torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0
                print(
                    f"Epoch {epoch:02d}/{NUM_EPOCHS} Phase{phase} Step {step:03d} | "
                    f"Loss:{loss_total.item():.4f} | "
                    f"PSNR-C:{pc:.2f} PSNR-S:{ps:.2f} | "
                    f"GPU:{mem:.0f}MB"
                )

            if device.type == "cuda" and step % 20 == 0:
                torch.cuda.empty_cache()

        avg_psnr_c = epoch_psnr_c / steps
        avg_psnr_s = epoch_psnr_s / steps
        print(f"=== Epoch {epoch}/{NUM_EPOCHS} done | Avg PSNR-C:{avg_psnr_c:.2f} | Avg PSNR-S:{avg_psnr_s:.2f} ===")

        improved = False
        if avg_psnr_s > best_psnr_s:
            best_psnr_s = avg_psnr_s
            improved = True

        ckpt = {
            "epoch": epoch,
            "phase": phase,
            "embedder": embedder.state_dict(),
            "extractor": extractor.state_dict(),
            "feat_extract": feat_ext.state_dict(),
            "enhance_pre": enh_pre.state_dict(),
            "enhance_post": enh_post.state_dict(),
            "best_psnr_s": best_psnr_s,
        }
        if epoch % 10 == 0:
            torch.save(ckpt, "models/checkpoints/latest.pth")
            print("Saved latest.pth")

        if improved:
            torch.save(ckpt, "models/checkpoints/best.pth")
            print(f"New best PSNR-S: {best_psnr_s:.2f} - saved best.pth")

        if epoch % 10 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(f"Current lr at epoch {epoch}: {current_lr:.6f}")

        scheduler.step()

    elapsed = (time.time() - start_time) / 60
    print(f"\nTraining complete in {elapsed:.1f} minutes")
    print(f"Final PSNR-C: {avg_psnr_c:.2f} | PSNR-S: {avg_psnr_s:.2f}")
    print("Run: python test.py --carrier data/sample_0.jpg --secret data/sample_1.jpg")


if __name__ == "__main__":
    main()
