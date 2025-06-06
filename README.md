<!-- README.md -->

## Real vs. AI-Generated Face Classifier

A PyTorch-based binary classification project that distinguishes between real human faces and AI-generated ones. Includes Grad-CAM interpretability, a Streamlit interface, and Docker-based deployment. Suitable for portfolio/CV.

---

### 🚀 Project Highlights (CV-Friendly)

- **Binary classification** using transfer learning (ResNet18/MobileNetV2) on real vs. AI faces.
- **Data pipeline**: Automated scripts for downloading and preprocessing CelebA (real) and “thispersondoesnotexist” (AI) images.
- **Model interpretability**: Integrated Grad-CAM to visualize model attention.
- **Deployment**: Interactive Streamlit web app with image upload and live inference.
- **Containerization**: Dockerized app for easy deployment.
- **CI/CD (optional)**: Sample GitHub Actions workflow for linting, unit tests, and Docker build.

---

### 📝 Table of Contents

1. [Installation](#installation)  
2. [Folder Structure](#folder-structure)  
3. [Dataset Acquisition & Preprocessing](#dataset-acquisition--preprocessing)  
4. [Model Training & Evaluation](#model-training--evaluation)  
5. [Grad-CAM Interpretability](#grad-cam-interpretability)  
6. [Web App Deployment (Streamlit)](#web-app-deployment-streamlit)  
7. [Docker Packaging](#docker-packaging)  
8. [(Optional) GitHub Actions CI/CD](#optional-github-actions-cicd)  
9. [Usage](#usage)  
10. [References](#references)

---

### Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<YOUR_USERNAME>/real-vs-ai-face-classifier.git
   cd real-vs-ai-face-classifier
