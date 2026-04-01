"""Data loading utilities for image watermarking."""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class WatermarkPairDataset(Dataset):
    """Creates (carrier, secret) image tensor pairs from one folder of images."""

    def __init__(self, image_dir: str, image_size: Tuple[int, int] = (224, 224)) -> None:
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.image_paths = self._collect_image_paths(self.image_dir)
        if len(self.image_paths) < 2:
            raise ValueError("At least two images are required to form carrier-secret pairs.")

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def _collect_image_paths(image_dir: Path) -> List[Path]:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sorted(
            [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions]
        )

    def __len__(self) -> int:
        # Pair images as (0,1), (2,3), ...
        return len(self.image_paths) // 2

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        carrier_path = self.image_paths[2 * index]
        secret_path = self.image_paths[2 * index + 1]

        carrier_image = Image.open(carrier_path).convert("RGB")
        secret_image = Image.open(secret_path).convert("RGB")

        carrier_tensor = self.transform(carrier_image)
        secret_tensor = self.transform(secret_image)

        return carrier_tensor, secret_tensor


def get_data_loader(
    image_dir: str,
    batch_size: int = 8,
    image_size: Tuple[int, int] = (224, 224),
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Returns a DataLoader that yields (carrier_image, secret_image) tensor batches."""

    dataset = WatermarkPairDataset(image_dir=image_dir, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
