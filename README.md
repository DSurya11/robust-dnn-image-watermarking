# Robust Image Watermarking — GAN + Dynamic Attention

## Results Summary
| Attack       | PSNR-C | PSNR-S | Result |
|---|---|---|---|
| Gaussian σ=1 | 28.56  | 19.03  | PASS   |
| Gaussian σ=10| 28.56  | 19.03  | PASS   |
| JPEG q=90    | 28.56  | 18.79  | PASS   |
| JPEG q=80    | 28.56  | 18.62  | PASS   |
| Round        | 28.56  | 19.02  | PASS   |

## Implementation Notes
This implementation focuses on the INN-based watermarking 
backbone with robustness training (Phases 1-2). The GAN 
adversarial training and Swin Transformer attention components 
from the paper are implemented in models/ but are experimental 
— the final evaluated model uses the Phase 1 INN checkpoint.

Training constraints vs paper:
- Hardware: CPU only (paper: RTX 4090)  
- Dataset: 79 synthetic images (paper: 800 DIV2K images)
- Epochs: ~200 (paper: 1600 epochs)

The 5 attack types all pass evaluation at PSNR-S > 18 dB.

## Improvement over baseline
All attacks improved by +14 dB PSNR-S (from 4.36 dB to ~19 dB)

## How to run
pip install -r requirements.txt
python train.py --phase 1 --epochs 50
python evaluate.py

## Modifications
- MOD1: Geometric attacks (rotation, crop, brightness)
- MOD2: Lambda hyperparameter tuning  
- MOD3: Watermark capacity analysis (noise/text/logo/natural)
