import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import random
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from model import FaceClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data/hard_finetune")
    parser.add_argument("--model-path", type=str, default="../models/best_model.pth")
    parser.add_argument("--output-path", type=str, default="../models/best_model_finetuned.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    return parser.parse_args()

class HardFTDataset(Dataset):
    def __init__(self, real_dir, ai_dir, real_transform, ai_transform):
        self.samples = []
        for path in os.listdir(real_dir):
            if path.lower().endswith((".jpg", ".png", ".jpeg")):
                self.samples.append((os.path.join(real_dir, path), 1))
        for path in os.listdir(ai_dir):
            if path.lower().endswith((".jpg", ".png", ".jpeg")):
                self.samples.append((os.path.join(ai_dir, path), 0))
        random.shuffle(self.samples)
        self.real_transform = real_transform
        self.ai_transform = ai_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        transform = self.real_transform if label == 1 else self.ai_transform
        return transform(image), label

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    real_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ai_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    real_dir = os.path.join(args.data_dir, "real")
    ai_dir   = os.path.join(args.data_dir, "ai")
    dataset = HardFTDataset(real_dir, ai_dir, real_transform, ai_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    print(f"Fine‐tuning dataset size: {len(dataset)} samples")

    model = FaceClassifier(backbone="resnet18")
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.train()

    # Fine-tune only the classifier head
    optimizer = Adam(model.classifier.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{args.epochs}  —  Loss: {avg_loss:.4f},  Acc: {acc:.2f}%")

    torch.save(model.state_dict(), args.output_path)
    print(f"✔️ Saved fine‐tuned model to {args.output_path}")

if __name__ == "__main__":
    main()
