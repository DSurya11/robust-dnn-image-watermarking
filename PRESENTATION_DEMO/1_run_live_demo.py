"""
LIVE DEMO SCRIPT — run this in front of the professor.
Shows real-time watermark embedding, attack, and extraction.
"""
import torch
from models.generator import SimpleISN
from utils.attacks import gaussian_noise, jpeg_compress, round_error
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as T
from PIL import Image
import os, sys

# Try relative path first (when run from project root)
CHECKPOINT = "models/checkpoints/phase1_best.pth"
# Fallback: checkpoint bundled with demo folder
if not os.path.exists(CHECKPOINT):
    CHECKPOINT = os.path.join(
        os.path.dirname(__file__),
        "models/checkpoints/phase1_best.pth"
    )
if not os.path.exists(CHECKPOINT):
    print("ERROR: Checkpoint not found.")
    print("Expected at: models/checkpoints/phase1_best.pth")
    exit(1)

DATA_DIR = "data/"

def run_demo():
    print("\n" + "="*60)
    print("  WATERMARKING DEMO — Live Evaluation")
    print("="*60)
    print("Input size: 128x128 (matches training and evaluation)")

    device = torch.device("cpu")
    isn = SimpleISN()
    ck = torch.load(CHECKPOINT, map_location=device)
    isn.load_state_dict(ck["isn_state_dict"])
    isn.eval()
    print(f"\nModel loaded. Best Val PSNR-S: {ck['psnr_s']:.2f} dB\n")

    imgs = [
        f
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ][:3]
    transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])

    attacks = [
        ("Gaussian σ=1",  lambda x: gaussian_noise(x, sigma=1)),
        ("Gaussian σ=10", lambda x: gaussian_noise(x, sigma=10)),
        ("JPEG q=90",     lambda x: jpeg_compress(x, quality=90)),
        ("JPEG q=80",     lambda x: jpeg_compress(x, quality=80)),
        ("Round",         round_error),
    ]

    print(f"{'Attack':<15} | {'PSNR-C':>8} | {'PSNR-S':>8} | {'Result':>6}")
    print("-" * 48)

    for attack_name, attack_fn in attacks:
        psnr_c_list, psnr_s_list = [], []
        for img_name in imgs:
            carrier = transform(Image.open(DATA_DIR + img_name).convert("RGB")).unsqueeze(0)
            secret_name = imgs[(imgs.index(img_name) + 1) % len(imgs)]
            secret  = transform(Image.open(DATA_DIR + secret_name).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                xc = isn.embed(carrier, secret)
                xd = attack_fn(xc)
                xe = isn.extract(xd)
            c = carrier.squeeze().permute(1,2,0).numpy()
            w = xc.squeeze().permute(1,2,0).numpy()
            s = secret.squeeze().permute(1,2,0).numpy()
            e = xe.squeeze().permute(1,2,0).numpy()
            psnr_c_list.append(psnr(c, w, data_range=1.0))
            psnr_s_list.append(psnr(s, e, data_range=1.0))
        pc = sum(psnr_c_list)/len(psnr_c_list)
        ps = sum(psnr_s_list)/len(psnr_s_list)
        status = "PASS" if ps > 18 else "FAIL"
        print(f"{attack_name:<15} | {pc:>8.2f} | {ps:>8.2f} | {status:>6}")

    print("-" * 48)
    print("\nDemo complete.")

if __name__ == "__main__":
    run_demo()
