import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_PATH = "/content/gdrive/MyDrive/WasteClassification/best_model.pth"

net = models.efficientnet_b3(weights=None)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 12)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.to(DEVICE)
net.eval()

print("✓ Model loaded successfully!")

CLASSES = [
    'Battery', 'Biological', 'Brown-Glass', 'Cardboard',
    'Clothes', 'Green-Glass', 'Metal', 'Paper', 'Plastic',
    'Shoes', 'Trash', 'White-Glass'
]

RECYCLABLE = {
    'Battery': False, 'Biological': False, 'Brown-Glass': True, 'Cardboard': True,
    'Clothes': True, 'Green-Glass': True, 'Metal': True, 'Paper': True,
    'Plastic': True, 'Shoes': False, 'Trash': False, 'White-Glass': True
}

BIODEGRADABLE = {
    'Battery': False, 'Biological': True, 'Brown-Glass': False, 'Cardboard': True,
    'Clothes': False, 'Green-Glass': False, 'Metal': False, 'Paper': True,
    'Plastic': False, 'Shoes': False, 'Trash': False, 'White-Glass': False
}

SUGGESTIONS = {
    "Battery": ["Drop at e-waste center.", "Do not throw in general waste.", "Store safely."],
    "Biological": ["Compost it.", "Put in organic waste bin.", "Use for soil enrichment."],
    "Plastic": ["Clean & recycle.", "Reuse as containers.", "Avoid burning."],
    "Paper": ["Recycle.", "Reuse for craft.", "Shred for packaging."],
    "Metal": ["Recycle at scrap center.", "Store safely.", "Reuse if possible."],
    "Trash": ["Put in general waste.", "Try reducing usage.", "Avoid mixing recyclable items."],
    "Clothes": ["Donate to the needy.", "Recycle the fabric.", "Upcycle for cleaning."],
    "Shoes": ["Donate wearable pairs.", "Recycle the sole material.", "Upcycle creatively."],
    "Cardboard": ["Flatten & recycle.", "Reuse for packing.", "Use for compost if clean."],
    "Green-Glass": ["Recycle.", "Reuse as containers.", "Handle carefully."],
    "White-Glass": ["Recycle.", "Use creatively in house for decorative purposes.", "Avoid throwing it in trash."],
    "Brown-Glass": ["Recycle.", "Reuse the bottle.", "Avoid breaking the glass."]
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
## heatmap generation
def generate_gradcam(image_tensor, model):
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor, targets=None)
    return grayscale_cam[0, :]


def apply_gradcam_to_image(original_image, gradcam):
    original_image = original_image.resize((224, 224))
    img_np = np.array(original_image) / 255.0

    if gradcam.shape != (224, 224):
        gradcam = cv2.resize(gradcam, (224, 224))

    visualization = show_cam_on_image(img_np, gradcam, use_rgb=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(visualization)
    ax.axis('off')

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('Model Focus Intensity', rotation=270, labelpad=20)

    return fig

def predict_waste(img):
    if img is None:
        return "<p style='color:red;'>⚠️ Please upload an image.</p>", None, None

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    display_img = cv2.resize(img, (224, 224))
    display_img = display_img.astype(np.float32) / 255.0

    with torch.no_grad():
        outputs = net(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    predicted_idx = torch.argmax(probs).item()
    predicted_label = CLASSES[predicted_idx]
    confidence = probs[predicted_idx].item() * 100

    THRESHOLD = 40

    if confidence < THRESHOLD:
        warning_html = f"""
        <div style="
            background: rgba(255, 87, 87, 0.12);
            border-left: 6px solid #ff3b3b;
            padding: 18px;
            border-radius: 10px;
            color: #ff4d4d;
            font-size: 1.2rem;
            line-height: 1.6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);
        ">
            <strong style="font-size:1.35rem; color:#ff5c5c;">
                ⚠️ Low Confidence Prediction
            </strong>

            <p style="margin-top:10px; color:#ff6d6d;">
                The model is only <strong>{confidence:.2f}%</strong> confident about this image.
            </p>

            <p style="color:#ff8080;">
                This does <strong>not</strong> appear to be a waste item.<br>
                Please upload a clearer image or ensure the object belongs to one of the 12 waste categories.
            </p>
        </div>
        """
        return warning_html, None, None



    prob_list = probs.cpu().numpy().tolist()

    recyclable = predicted_label in ["Glass", "Metal", "Paper", "Plastic"]
    biodegradable = predicted_label in ["Biological"]

    suggestion_list = SUGGESTIONS.get(predicted_label, ["No suggestions available."])
    suggestions_html = "".join([f"<li>{s}</li>" for s in suggestion_list])

    result_html = f"""
    <div style="font-size:1.15rem; line-height:1.6;">
        <p><strong>🎯 Predicted:</strong> {predicted_label}</p>
        <p><strong>📊 Confidence:</strong> {confidence:.2f}%</p>
        <p><strong>♻️ Recyclable:</strong> {"YES" if recyclable else "NO"}</p>
        <p><strong>🌱 Biodegradable:</strong> {"YES" if biodegradable else "NO"}</p>
        <hr style="margin:15px 0; opacity:0.3;">
  <h3 style="margin-bottom:8px;">💡 Disposal Suggestions</h3>
  <ul style="margin-left:20px; font-size:1.1rem;">
      {suggestions_html}
  </ul>
    </div>
    """

    def generate_gradcam(image_tensor, model):
        target_layer = model.features[-1]
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=image_tensor)
        return grayscale_cam[0]

    gradcam_mask = generate_gradcam(img_tensor, net)

    heatmap = cv2.applyColorMap((gradcam_mask * 255).astype(np.uint8),
                                cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    gradcam_img = (0.4 * heatmap + 0.6 * (display_img * 255)).astype(np.uint8)

    fig_conf, ax = plt.subplots()
    ax.bar(CLASSES, prob_list)
    ax.set_ylabel("Probability")
    ax.set_title("Model Confidence Distribution")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    fig_cam, ax2 = plt.subplots()

    im = ax2.imshow(gradcam_img)
    ax2.axis("off")
    ax2.set_title("Grad-CAM Heatmap")

    cmap = mpl.cm.jet
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cbar = fig_cam.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax2,
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label("Attention Intensity (Blue → Red)", rotation=270, labelpad=15)

    return result_html, fig_conf, fig_cam




with gr.Blocks(title="TrashScan: Smart Waste Classifier", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <style>

    .main-container {
        max-width: 60%;
        margin: auto;
    }

    #project_title {
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 5px;
        color: #00e676 !important;
    }

    #project_subtitle {
        text-align: center;
        margin-bottom: 40px;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border-radius: 14px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 25px;
    }

    .result-card-content {
        font-size: 1.05rem;
        line-height: 1.55;
    }

    </style>
    """)

    gr.HTML("""
    <h1 id='project_title'>♻️ TrashScan: Smart Waste Classifier</h1>
    <p id='project_subtitle'>Click. Scan. Dispose Right.</p>
    """)

    with gr.Column(elem_classes="main-container"):

        with gr.Column(elem_classes="card"):
            gr.Markdown("### 📸 Upload")
            image_input = gr.Image(type="numpy", height=350)
            classify_btn = gr.Button("🔍 Classify", variant="primary")

        with gr.Column(elem_classes="card"):
            gr.Markdown("### 🧾 Results")
            result_output = gr.HTML(elem_classes="result-card-content")

        with gr.Column(elem_classes="card"):
            gr.Markdown("### 📈 Confidence")
            chart_output = gr.Plot()

        with gr.Column(elem_classes="card"):
            gr.Markdown("### 🔥 Grad-CAM Visualization")
            gradcam_output = gr.Plot()

    classify_btn.click(
        fn=predict_waste,
        inputs=[image_input],
        outputs=[result_output, chart_output, gradcam_output]
    )

    def clear_outputs(img):
        if img is None:
            return "", None, None
        return gr.update(), gr.update(), gr.update()

    image_input.change(
        fn=clear_outputs,
        inputs=[image_input],
        outputs=[result_output, chart_output, gradcam_output]
    )


print("✓ Launching app…")
demo.launch(share=True, show_error=True)
