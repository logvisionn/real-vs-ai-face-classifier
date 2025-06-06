import os
# ── Windows OpenMP workaround ────────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st

# ── set_page_config must be the very first Streamlit call ─────────────────────
st.set_page_config(
    page_title="Real vs AI Face Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from src.model import FaceClassifier
from src.gradcam import generate_gradcam_heatmap, overlay_heatmap_on_image

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"

# Use exactly the same transforms as in training / the notebook:
# e.g. Resize(256) + CenterCrop(224), then ToTensor + Normalize
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ── 2. LOAD MODEL ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = FaceClassifier(backbone="resnet18")
    # Use weights_only=True to silence the FutureWarning
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model

model = load_model()

# Hook exactly the *last* BasicBlock in layer4 (so Captum sees the final feature map)
target_layer = model.backbone.layer4[-1]

# ── 3. PREPROCESSING ──────────────────────────────────────────────────────────

def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

# ── 4. STREAMLIT UI ───────────────────────────────────────────────────────────

st.title("Real vs AI Face Classifier")
st.write("Upload a face image—model predicts Real vs. AI and shows Grad‐CAM.")

uploaded_file = st.file_uploader("Choose an image…", type=["jpg", "jpeg", "png"])
run_cam = st.checkbox("Show Grad‐CAM overlay", value=True)

if uploaded_file is not None:
    st.success("File uploaded! Running inference…")

    # 1) Display uploaded image at full container width
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("")

    # 2) Preprocess & forward
    input_tensor = preprocess_image(image)  # [1,3,224,224]
    with torch.no_grad():
        logits = model(input_tensor)                # [1,2]
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        class_idx = int(np.argmax(probs))           # 0 or 1
        confidence = float(probs[class_idx])

    # 3) Correct label mapping (0=AI, 1=Real)
    if class_idx == 0:
        label = "AI-Generated"
    else:
        label = "Real"

    st.markdown(f"### Prediction: **{label}**  —  confidence: {confidence*100:.2f}%")

    # 4) Grad-CAM: only if checkbox is checked
    if run_cam:
        st.write("🔍 Generating Grad‐CAM overlay…")
        # This returns a 224×224 float map in [0,1], exactly as in notebook
        heatmap = generate_gradcam_heatmap(
            model=model,
            target_layer=target_layer,
            input_tensor=input_tensor,
            target_class=class_idx,
            device=DEVICE
        )

        # Blend & colorize exactly as in the notebook’s upsample_and_overlay
        overlay_np = overlay_heatmap_on_image(image, heatmap, alpha=0.5)

        # Display the overlay at full container width so it matches the original’s size
        st.image(overlay_np, caption="Grad-CAM Overlay", use_container_width=True)

