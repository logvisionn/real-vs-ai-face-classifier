import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import FaceClassifier
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

def load_image(img_path: str, device: torch.device, size=(224,224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    return img, tensor

def generate_gradcam(model: torch.nn.Module, img_tensor: torch.Tensor,
                     target_layer: str = "layer4", target_class: int = None):
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
    out = model(img_tensor)
    if target_class is None:
        target_class = int(torch.argmax(out, dim=1).item())
    activation_map = cam_extractor(target_class, out.squeeze(0))
    cam_map = activation_map[target_layer].squeeze().cpu().numpy()
    return cam_map, target_class

def visualize_gradcam(img: Image.Image, cam_map: np.ndarray, output_path: str = None):
    img_np = np.array(img)
    heatmap_img = overlay_mask(img_np, cam_map, alpha=0.5)
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap_img)
    plt.axis("off")
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a sample image.")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth")
    parser.add_argument("--img-path", type=str, required=True, help="Path to a single image.")
    parser.add_argument("--output-path", type=str, default=None, help="Output for the heatmap overlay.")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--target-layer", type=str, default="layer4")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceClassifier(backbone=args.backbone, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    orig_img, img_tensor = load_image(args.img_path, device, size=(224,224))
    cam_map, predicted_class = generate_gradcam(model, img_tensor, target_layer=args.target_layer)
    classes = ["ai", "real"]
    print(f"Predicted class: {classes[predicted_class]}")
    visualize_gradcam(orig_img, cam_map, args.output_path)
