from pathlib import Path

from PIL import Image, ImageStat


def main() -> None:
    data_dir = Path(__file__).resolve().parent

    image_paths = sorted(
        [
            p
            for p in data_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".png"}
            and p.suffix.lower() != ".py"
            and p.name != "prepare_data.py"
        ]
    )

    prepared_files: list[Path] = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            processed = img.convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
            processed.save(image_path, format="JPEG", quality=95)
            mean_rgb = ImageStat.Stat(processed).mean
            mean_value = sum(mean_rgb) / 3.0
            print(f"Prepared: {image_path.name} | size=128x128 | mean={mean_value:.2f}")
        prepared_files.append(image_path)

    print(f"Total prepared: {len(prepared_files)}")

    all_ok = True
    for image_path in prepared_files:
        with Image.open(image_path) as img:
            if img.convert("RGB").size != (128, 128):
                all_ok = False

    if all_ok:
        print("DATA OK")
    else:
        print("DATA CHECK FAILED")


if __name__ == "__main__":
    main()
