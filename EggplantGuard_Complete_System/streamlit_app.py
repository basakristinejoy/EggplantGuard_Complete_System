from pathlib import Path

import streamlit as st
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "best.pt"

CLASS_LABELS = [
    "Healthy Leaf",
    "Leaf Miner Damage",
    "Fresh-Market-Ready Fruit",
    "Damaged-Borer-Infested Fruit",
]

RECOMMENDATIONS = {
    "Healthy Leaf": "Healthy eggplant foliage. Continue regular monitoring and maintain crop hygiene.",
    "Leaf Miner Damage": "Leaf miner symptoms detected. Remove damaged leaves and use integrated pest management controls.",
    "Fresh-Market-Ready Fruit": "Fruit appears fresh and market-ready. Maintain proper harvest handling and storage practices.",
    "Damaged-Borer-Infested Fruit": "Borer damage suspected. Remove infested fruit and apply appropriate control measures immediately.",
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.warning(
            "No trained model was found at model/best.pt. Train the YOLO model first, then restart the app."
        )
        return None

    model = YOLO(str(MODEL_PATH))
    return model


def predict_image(model, uploaded_file):
    image_bytes = uploaded_file.read()
    temp_path = BASE_DIR / "tmp_uploaded_image.jpg"
    temp_path.write_bytes(image_bytes)

    try:
        results = model(str(temp_path), verbose=False)
        result = results[0]
        probs = result.probs

        if probs is None:
            return None, "Model did not return class probabilities."

        idx = int(probs.top1)
        label = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else str(idx)
        confidence = float(probs.top1conf * 100)
        return label, confidence
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


st.set_page_config(page_title="EggplantGuard", page_icon="🍆", layout="wide")

st.title("🍆 EggplantGuard")
st.caption("Upload an eggplant leaf or fruit image to classify it using the trained YOLO model.")

model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = uploaded_file.read()
    st.image(image, caption="Uploaded image", use_column_width=True)

    if model is None:
        st.info("Train the model with train_yolo.py before running prediction.")
    else:
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image(model, uploaded_file)

        if label is None:
            st.error("Prediction failed. Please try a clearer image.")
        else:
            st.success(f"Prediction: {label}")
            st.metric("Confidence", f"{confidence:.2f}%")
            st.write(RECOMMENDATIONS.get(label, "Please inspect the crop and use good agronomic practices."))

st.sidebar.header("Project info")
st.sidebar.write("This Streamlit app is a deployment-friendly version of the EggplantGuard project.")
st.sidebar.write("Model file expected: model/best.pt")
st.sidebar.write("Use train_yolo.py to train the model before deployment.")
