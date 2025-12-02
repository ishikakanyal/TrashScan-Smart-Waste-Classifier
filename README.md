# 🗑️ TrashScan: Smart Waste Classifier

TrashScan is an intelligent waste classification system using deep learning and transfer learning to identify different types of waste such as **plastic, metal, glass, paper, and organic waste**. The system is optimized for real-world use and includes **Grad-CAM visualizations** for explainability.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Visualization & Explainability](#visualization--explainability)
- [Model Download](#model-download)
- [Installation](#installation)
- [Usage](#usage)

---

## 📖 Overview
TrashScan leverages a **fine-tuned EfficientNet model** to classify waste images. It provides a prediction along with a **confidence score**, and includes **Grad-CAM heatmaps** to highlight the regions that influenced the prediction.

This system is designed for **web** or **mobile deployment**, making it ideal for smart recycling, waste management, and educational applications.

---

## ⭐ Features
- 🔍 Classifies waste into:
  - **Plastic**
  - **Metal**
  - **Glass**
  - **Paper**
  - **Organic**
- 📊 Provides **confidence scores**
- 🔥 Offers **Grad-CAM heatmap visualizations** for interpretability
- ⚡ Fast and lightweight — suitable for deployment
- 🌐 Runs on a **Gradio web interface**

---

## 🔄 Project Workflow
1. **Data Collection:** Gather labeled images of waste categories  
2. **Preprocessing:** Resize, normalize, and augment images for robustness  
3. **Model Training:** Fine-tune EfficientNet on the custom dataset  
4. **Evaluation:** Assess accuracy, precision, recall, and F1-score  
5. **Explainability:** Generate Grad-CAM heatmaps  
6. **Deployment:** Run via a user-friendly Gradio interface  

---

## 🧠 Model Architecture
- **Base Model:** EfficientNet (pre-trained on ImageNet)  
- **Input Size:** `224 × 224 × 3`  
- **Output:** Softmax across waste categories  
- **Added Layers:**  
  - Dense layer(s) with ReLU  
  - Dropout for regularization  
- **Optimizer:** Adam  
- **Loss Function:** Cross-entropy  

---

## 🔥 Visualization & Explainability
TrashScan uses **Grad-CAM** to visualize what parts of the image influenced the model's decision.

- Highlights important regions  
- Helps validate model reasoning  
- Useful for debugging and transparency  

Users see:
- A **heatmap**
- Overlaid visualization on the original image  
- Predicted class + confidence score  

---

## 📥 Model Download
Because the trained model file is large, it is hosted externally.

👉 **Download the trained model from Google Drive:**  
  [Click here to download](https://drive.google.com/file/d/1sPDiy_5dKFEzvPo_RgJOQeynHl9aCXcz/view?usp=drive_link)

After downloading, place the model file in your project directory:

models/

└── trashscan_model.pth


---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/ishikakanyal/TrashScan-Smart-Waste-Classifier.git
cd TrashScan-Smart-Waste-Classifier
2. Install dependencies
pip install -r requirements.txt
```
### 🧩 Requirements
- Python 3.8+
- torch  
- torchvision  
- gradio  
- matplotlib  
- numpy  

---

## 🚀 Usage

### Start the Gradio web app:
```bash
python app.py
```
### Steps to Use TrashScan

1. Upload an image of waste.
2. Get the predicted class along with the confidence score.
3. View the Grad-CAM visualization highlighting important regions.
4. Make sure the model file is placed correctly in the project folder before running the app.

## 🙏 Acknowledgements
- This project uses [EfficientNet](https://arxiv.org/abs/1905.11946) for transfer learning.
- Grad-CAM implementation inspired by [this tutorial](https://arxiv.org/abs/1610.02391).

