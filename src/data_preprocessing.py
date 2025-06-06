import os
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_dataset(real_dir: str, ai_dir: str, output_dir: str,
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, size=(224,224)):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
    ai_images   = [os.path.join(ai_dir, f) for f in os.listdir(ai_dir)]
    real_labels = ["real"] * len(real_images)
    ai_labels   = ["ai"]   * len(ai_images)

    all_paths  = real_images + ai_images
    all_labels = real_labels  + ai_labels

    paths_train, paths_temp, labels_train, labels_temp = train_test_split(
        all_paths, all_labels, stratify=all_labels, test_size=(1.0 - train_ratio), random_state=42
    )
    paths_val, paths_test, labels_val, labels_test = train_test_split(
        paths_temp, labels_temp, stratify=labels_temp,
        test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
    )

    splits = {
        "train": (paths_train, labels_train),
        "val":   (paths_val,   labels_val),
        "test":  (paths_test,  labels_test),
    }

    for split_name, (paths, labels) in splits.items():
        dest_root = os.path.join(output_dir, split_name)
        for path, label in tqdm(zip(paths, labels), desc=f"Copying to {split_name}", total=len(paths)):
            dest_folder = os.path.join(dest_root, label)
            os.makedirs(dest_folder, exist_ok=True)
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img = img.resize(size, Image.BILINEAR)
                    fname = os.path.basename(path)
                    img.save(os.path.join(dest_folder, fname))
            except Exception:
                continue

def main(args):
    os.makedirs(args.processed_dir, exist_ok=True)
    assert os.path.isdir(args.real_raw), f"Real raw folder not found: {args.real_raw}"
    assert os.path.isdir(args.ai_raw),   f"AI raw folder not found: {args.ai_raw}"
    split_dataset(
        real_dir=args.real_raw,
        ai_dir=args.ai_raw,
        output_dir=args.processed_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        size=(args.width, args.height),
    )
    print("Data preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and split dataset.")
    parser.add_argument("--real-raw", type=str, default="data/raw/real", help="Path to raw real faces.")
    parser.add_argument("--ai-raw",   type=str, default="data/raw/ai",   help="Path to raw AI faces.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Output directory for processed data.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val-ratio",   type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio",  type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--width", type=int, default=224, help="Target width.")
    parser.add_argument("--height", type=int, default=224, help="Target height.")
    args = parser.parse_args()
    main(args)
