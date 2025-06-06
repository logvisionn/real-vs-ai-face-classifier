import torch
import numpy as np
import cv2
from captum.attr import LayerGradCam
from PIL import Image

def generate_gradcam_heatmap(model, target_layer, input_tensor, target_class, device="cpu"):
    """
    Generates a Grad‐CAM heatmap (normalized & upsampled to 224×224) for a single image tensor,
    by hooking only the *last* BasicBlock in ResNet-18’s layer4 (i.e. layer4[-1]).

    Args:
      model (torch.nn.Module):      The trained classifier (in eval mode).
      target_layer (nn.Module):     The last BasicBlock, e.g. model.backbone.layer4[-1].
      input_tensor (torch.Tensor):  A single image [1,3,224,224], already normalized.
      target_class (int):           0 or 1, the predicted class index.
      device (str or torch.device): “cpu” or “cuda”.

    Returns:
      heatmap_up (np.ndarray):
        A 2D NumPy array shape [224,224], float values ∈ [0,1], the Grad-CAM map.
    """

    # 1) Move model + input to device, ensure eval mode
    model = model.to(device).eval()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)

    # 2) Create a LayerGradCam that hooks only the final BasicBlock in layer4
    cam_extractor = LayerGradCam(model, target_layer)

    # 3) Forward pass (registers forward/backward hooks inside Captum)
    _ = model(input_tensor)  # -> [1, num_classes]

    # 4) Wrap target_class into a single‐element tensor (Captum expects a tensor)
    target_tensor = torch.tensor([target_class], device=device)

    # 5) Compute the raw Grad-CAM attributions; result shape is [1,1,7,7]
    cam_attributions = cam_extractor.attribute(input_tensor, target=target_tensor)

    # 6) Squeeze to get a 2D array [7,7]
    raw_heat = cam_attributions.squeeze().detach().cpu().numpy()  # shape [7,7]

    # 7) Upsample the [7×7] raw heat to [224×224] with bilinear interpolation
    heat_up = cv2.resize(
        raw_heat,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    # 8) Normalize the upsampled map to [0,1]
    min_val, max_val = float(heat_up.min()), float(heat_up.max())
    if (max_val - min_val) > 1e-8:
        heat_up = (heat_up - min_val) / (max_val - min_val)
    else:
        heat_up = np.zeros_like(heat_up)

    # 9) Manually remove all registered hooks so they don’t accumulate on subsequent calls
    if hasattr(cam_extractor, "_handles"):
        for handle in cam_extractor._handles:
            handle.remove()
        cam_extractor._handles = []

    return heat_up  # shape: [224,224], floats in [0,1]


def overlay_heatmap_on_image(pil_image, heatmap, alpha=0.5):
    """
    Overlays a Grad-CAM heatmap (224×224 float [0,1]) onto a PIL image,
    using the exact same steps as in 02-gradcam-demo.ipynb:
     1) Resize original to 224×224, convert to NumPy
     2) Convert heatmap [0,1] → [0,255] uint8
     3) cv2.applyColorMap → BGR
     4) cv2.cvtColor to RGB
     5) cv2.addWeighted(original, 0.5, heat_color, 0.5, 0)

    Args:
      pil_image (PIL.Image.Image): The original RGB image (any size).
      heatmap (np.ndarray):        2D float array [224,224] in [0,1].
      alpha (float):               Blend factor (0.0=image only, 1.0=heatmap only).

    Returns:
      overlay_rgb (np.ndarray):
        A 3-channel uint8 RGB array [224,224,3] showing the blended result.
    """
    # 1) Resize original to 224×224, convert to NumPy (uint8)
    face = np.array(pil_image.convert("RGB").resize((224, 224)))  # shape [224,224,3]

    # 2) Convert normalized [0,1] float → [0,255] uint8
    heatmap_uint8 = np.uint8(heatmap * 255)

    # 3) Apply JET colormap (yields BGR)
    heat_col_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 4) Convert BGR → RGB
    heat_col_rgb = cv2.cvtColor(heat_col_bgr, cv2.COLOR_BGR2RGB)

    # 5) Blend 50/50: original + heatmap
    overlay = cv2.addWeighted(face, 1.0 - alpha, heat_col_rgb, alpha, 0)

    return overlay  # shape [224,224,3], dtype=uint8 (values 0–255)

if __name__ == "__main__":
    import argparse
    from torchvision import transforms
    from src.model import FaceClassifier  # adjust import if needed

    parser = argparse.ArgumentParser(description="Generate & save a Grad-CAM overlay.")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the trained `.pth` file (e.g. models/best_model.pth)."
    )
    parser.add_argument(
        "--image-path", type=str, required=True,
        help="Path to a single RGB face image (JPG/PNG)."
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Where to save the final overlay (PNG)."
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet18", choices=["resnet18","mobilenet_v2"],
        help="Which backbone was used in training (to rebuild the same model architecture)."
    )
    parser.add_argument(
        "--class-index", type=int, default=1,
        help="0 = class ‘ai’, 1 = class ‘real’ (use whichever you want to visualize)."
    )
    parser.add_argument(
        "--use-gpu", action="store_true",
        help="If set, run Grad-CAM on GPU. Otherwise uses CPU."
    )
    args = parser.parse_args()

    # 1) Device selection
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2) Load the model
    model = FaceClassifier(backbone=args.backbone)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    # 3) Identify the target layer
    #    Here we assume the “layer4” block in ResNet‐18; for a different backbone adjust accordingly.
    target_layer = model.backbone.layer4

    # 4) Preprocess the input image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    pil_img = Image.open(args.image_path).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0)  # [1, 3, 224, 224]

    # 5) Generate heatmap
    heatmap = generate_gradcam_heatmap(
        model=model,
        target_layer=target_layer,
        input_tensor=input_tensor,
        target_class=args.class_index,
        device=device
    )

    # 6) Overlay heatmap on PIL image
    overlay = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.5)

    # 7) Save result
    #    overlay is an H×W×3 NumPy array in RGB
    overlay_pil = Image.fromarray(overlay)
    overlay_pil.save(args.output_path)
    print(f"Saved Grad-CAM overlay to: {args.output_path}")
