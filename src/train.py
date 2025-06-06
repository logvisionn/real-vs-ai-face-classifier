# ---------------------------------------------------------
# train.py  (fixed to suppress OpenMP duplication warnings)
# ---------------------------------------------------------

import os
# ─── Suppress OpenMP “duplicate lib” error ─────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Optional: if you still see OpenMP issues, you can force single‐worker:
# os.environ["OMP_NUM_THREADS"] = "1"
# ──────────────────────────────────────────────────────────────

import copy
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from model import FaceClassifier

def get_dataloaders(data_dir: str, batch_size: int, num_workers: int = 4):
    image_size = 224

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
        for x in ["train", "val", "test"]
    }

    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=(x == "train"),
            num_workers=num_workers  # if you still get OMP errors, set num_workers=0
        )
        for x in ["train", "val", "test"]
    }

    class_names = image_datasets["train"].classes
    return dataloaders, class_names

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available?", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("No GPU found, training on CPU")

    dataloaders, class_names = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    print(f"Classes: {class_names}")

    # Instantiate FaceClassifier
    model = FaceClassifier(backbone=args.backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = accuracy_score(all_labels, all_preds)
            epoch_f1  = f1_score(all_labels, all_preds, average="weighted")

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")

            # Save best‐on‐validation weights
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        print()

    print(f"Best val Acc: {best_acc:.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "best_model.pth")
    torch.save(best_model_wts, save_path)
    print(f"Saved best model to {save_path}")

    # Load best weights into model for final return (so evaluation uses best‐on‐val)
    model.load_state_dict(best_model_wts)
    return model, class_names

def evaluate_model(model, dataloader, device):
    model.eval().to(device)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float("nan")

    print(f"Test Acc: {acc:.4f} F1: {f1:.4f} ROC_AUC: {roc_auc:.4f}")
    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a real vs AI face classifier.")
    parser.add_argument(
        "--data-dir", type=str, default="data/processed",
        help="Processed data directory (train/val/test inside)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Where to save best model (.pth)."
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet18",
        choices=["resnet18", "mobilenet_v2"],
        help="Which backbone to use."
    )
    parser.add_argument(
        "--num-epochs", type=int, default=10,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for train/val/test."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate for Adam."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader num_workers (set to 0 if you get OMP errors)."
    )
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="If set, skip training and only run evaluation on test set."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.evaluate_only:
        model = FaceClassifier(backbone=args.backbone)
        ckpt_path = os.path.join(args.output_dir, "best_model.pth")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)

        dataloaders, class_names = get_dataloaders(
            args.data_dir, args.batch_size, args.num_workers
        )
        test_loader = dataloaders["test"]
        evaluate_model(model, test_loader, device)
        exit()

    model, class_names = train_model(args)

    # After training, run final test evaluation:
    dataloaders, _ = get_dataloaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    test_loader = dataloaders["test"]
    evaluate_model(model, test_loader, device)
