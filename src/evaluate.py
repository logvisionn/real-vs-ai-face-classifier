import os
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model import FaceClassifier

def get_test_loader(data_dir: str, batch_size: int, num_workers: int = 4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceClassifier(backbone=args.backbone)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    test_loader = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
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

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Processed data dir")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth", help="Path to saved model.")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","mobilenet_v2"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
