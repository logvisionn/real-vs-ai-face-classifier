# test_harness_gui.py

import os
import csv
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Adjust this import if needed to match your folder structure
from src.model import FaceClassifier

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Device (CPU or GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to your checkpoint
MODEL_PATH = Path("models") / "best_model.pth"

# Preprocessing parameters (must match training)
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Output CSV file
OUTPUT_CSV = Path("test_results.csv")

# ── UTILITY FUNCTIONS ─────────────────────────────────────────────────────────

def load_model():
    """
    Load the trained FaceClassifier from MODEL_PATH into evaluation mode.
    """
    if not MODEL_PATH.exists():
        print(f"ERROR: Could not find model at {MODEL_PATH}")
        sys.exit(1)

    model = FaceClassifier(backbone="resnet18")
    # Suppress the FutureWarning by specifying weights_only=True
    state_dict = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

def preprocess_image(img_path: Path) -> torch.Tensor:
    """
    Given a path to an image file, open it with PIL, resize to IMG_SIZE,
    convert to tensor, and normalize. Returns a [1,3,224,224] tensor on DEVICE.
    """
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return tensor

def infer(model, input_tensor: torch.Tensor) -> (str, float):
    """
    Run the model on a single preprocessed image tensor.
    Returns (predicted_label, confidence), where predicted_label ∈ {"Real", "AI-Generated"}.
    """
    with torch.no_grad():
        logits = model(input_tensor)                 # [1, 2]
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # [prob_real, prob_ai]
        class_idx = int(probs.argmax())              # 0 or 1
        confidence = float(probs[class_idx])         # e.g., 0.9876

    label = "AI-Generated" if class_idx == 1 else "Real"
    return label, confidence

def init_csv(path: Path):
    """
    If the CSV does not exist, create it and write the header row.
    Otherwise, do nothing.
    """
    if not path.exists():
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "predicted_label", "confidence", "actual_label"])

def append_result(path: Path, image_path: str, pred: str, conf: float, actual: str):
    """
    Append a single row to the CSV at `path`.
    """
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([image_path, pred, f"{conf:.4f}", actual])

def select_image_file() -> Path:
    """
    Opens a file‐dialog window for the user to select an image (jpg/jpeg/png).
    Returns a Path object or None if the user cancels.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main Tk window
    root.update()
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
    )
    root.destroy()
    if not file_path:
        return None
    return Path(file_path)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    print("\n=============================")
    print("  Real vs. AI Test Harness  ")
    print("=============================\n")

    # 1) Load model once
    model = load_model()
    print(f"Loaded model from {MODEL_PATH} onto {DEVICE}.\n")

    # 2) Ensure CSV is ready
    init_csv(OUTPUT_CSV)
    print(f"Test results (if any) will be saved to: {OUTPUT_CSV}\n")

    print("Instructions:")
    print(" • Click the file‐picker dialog to choose an image.")
    print(" • The script will output the model’s prediction and confidence.")
    print(" • Then you will be prompted for the actual label: 'r' for real, 'f' for fake.")
    print(" • Selecting “Cancel” in the dialog will quit the program.\n")

    while True:
        # 3) Open file dialog so user picks an image
        print("Opening file dialog... (Cancel to quit)")
        img_path = select_image_file()
        if img_path is None:
            # User hit “Cancel”
            print("\nQuitting. Thank you for testing!\n")
            break

        # Validate the extension just in case
        if not img_path.exists() or img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            print(f"❌ Invalid file selected: {img_path}\n")
            # Continue to next iteration (reopen dialog)
            continue

        try:
            # 4) Preprocess & infer
            input_tensor = preprocess_image(img_path)
            predicted_label, confidence = infer(model, input_tensor)
            print(f"\n✅ Prediction: {predicted_label} (confidence: {confidence*100:.2f} %)")
        except Exception as e:
            print(f"❌ Error during inference: {e}\n")
            continue

        # 5) Ask for ground‐truth
        while True:
            gt = input("Enter actual label ('r' for real, 'f' for fake): ").strip().lower()
            if gt in ("r", "f"):
                actual_label = "Real" if gt == "r" else "AI-Generated"
                break
            else:
                print("  ▶️ Invalid input. Please type 'r' or 'f'.")

        # 6) Save result to CSV
        append_result(OUTPUT_CSV, str(img_path), predicted_label, confidence, actual_label)
        print(f"✔️ Logged: [{img_path}], pred={predicted_label}, conf={confidence:.4f}, true={actual_label}\n")

    print(f"Test results have been written/appended to: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()
