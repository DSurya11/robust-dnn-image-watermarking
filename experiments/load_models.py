"""Shared checkpoint loader for all experiments."""

import os
import sys


def load_all_models(device, checkpoint_path="models/checkpoints/latest.pth"):
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from models.embedder import WatermarkEmbedder
    from models.extractor import WatermarkExtractor
    from models.discriminator import Discriminator
    from models.feat_extract import DifferentialFeatureExtractor
    from models.enhance import EnhanceModule

    embedder = WatermarkEmbedder().to(device)
    extractor = WatermarkExtractor().to(device)
    disc = Discriminator().to(device)
    feat_ext = DifferentialFeatureExtractor().to(device)
    enh_pre = EnhanceModule(window_size=4).to(device)
    enh_post = EnhanceModule(window_size=8).to(device)

    selected_checkpoint = None
    candidates = ["models/checkpoints/best.pth", "models/checkpoints/latest.pth"]
    if checkpoint_path not in candidates:
        candidates.insert(0, checkpoint_path)

    for candidate in candidates:
        if os.path.exists(candidate):
            selected_checkpoint = candidate
            break

    if selected_checkpoint is not None:
        ckpt = torch.load(selected_checkpoint, map_location=device)
        embedder.load_state_dict(ckpt["embedder"])
        extractor.load_state_dict(ckpt["extractor"])
        if "discriminator" in ckpt:
            disc.load_state_dict(ckpt["discriminator"])
        feat_ext.load_state_dict(ckpt["feat_extract"])
        enh_pre.load_state_dict(ckpt["enhance_pre"])
        enh_post.load_state_dict(ckpt["enhance_post"])
        print(f"Loaded checkpoint: {selected_checkpoint}")
    else:
        print("WARNING: No checkpoint found (best.pth/latest.pth). Using random weights.")

    for m in [embedder, extractor, disc, feat_ext, enh_pre, enh_post]:
        m.eval()

    return embedder, extractor, disc, feat_ext, enh_pre, enh_post
