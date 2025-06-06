import os
import torch
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from model import FaceClassifier
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

@st.cache_resource(show_spinner=False)
def load_model(model_path: str, backbone: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceClassifier(backbone=backbone, pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device

def preprocess_image(image: Image.Image, size=(224,224)):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def get_prediction(model, device, img_tensor: torch.Tensor):
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred_idx = np.argmax(probs)
    classes = ["ai", "real"]
    return classes[pred_idx], float(probs[pred_idx]), logits

def generate_cam(model, device, img_tensor, target_layer="layer4"):
    cam_extractor = SmoothGradCAMpp(model, target_layer=target_layer)
    outputs = model(img_tensor.to(device))
    target_class = int(torch.argmax(outputs, dim=1).item())
    activation_map = cam_extractor(target_class, outputs.squeeze(0))
    cam_map = activation_map[target_layer].squeeze().cpu().numpy()
    return cam_map

def overlay_heatmap(original_image: Image.Image, cam_map: np.ndarray, alpha=0.5):
    orig_np = np.array(original_image)
    heatmap_overlay = overlay_mask(orig_np, cam_map, alpha=alpha)
    return cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)

st.set_page_config(page_title="Real vs AI Face Classifier", layout="centered")

st.title("🕵️‍♂️ Real vs. AI-Generated Face Classifier")
st.markdown("""
Upload a face image below. The model will predict whether it’s real or AI-generated.  
Optionally, view Grad-CAM heatmap to see which regions influenced the decision.
""")

st.sidebar.header("Settings")
backbone = st.sidebar.selectbox("Backbone", options=["resnet18", "mobilenet_v2"])
show_cam = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/best_model.pth"))
model, device = load_model(model_path, backbone)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = preprocess_image(image)
    label, confidence, _ = get_prediction(model, device, img_tensor)
    st.markdown(f"### Prediction: **{label.upper()}** (Confidence: {confidence*100:.1f}%)")

    if show_cam:
        st.markdown("#### Grad-CAM Heatmap")
        cam_map = generate_cam(model, device, img_tensor, target_layer="layer4")
        heatmap_img = overlay_heatmap(image, cam_map)
        st.image(heatmap_img, channels="BGR", use_column_width=True)
