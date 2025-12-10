import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

# ---- Load Model ----
model = YOLO("best.pt")
class_names = model.names

# ---- Page Header ----
st.title("🧴 Acne Detection Model")
st.markdown(
    """
    <div style="text-align: center; font-size:18px; color:#444;">
    This <b>AI-powered dermatology assistant</b> uses advanced computer vision techniques to 
    accurately classify <b>16 types of acne and related skin conditions</b>. 
    It provides fast, reliable, and easy-to-interpret results to help dermatologists 
    and individuals make informed decisions about skin health.
    </div>
    """,
    unsafe_allow_html=True
)

# ---- Image Upload ----
uploaded_file = st.file_uploader("📤 Upload a skin image for analysis", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # ---- Prediction ----
    with st.spinner("⏳ Analyzing image..."):
        time.sleep(1)
        results = model.predict(image)

    # ---- Probabilities ----
    probs = results[0].probs.data.cpu().numpy()

    # ---- Extract Top Prediction ----
    top_idx = probs.argmax()
    top_label = class_names[top_idx]
    top_conf = probs[top_idx] * 100

    # ---- Display Result Card ----
    card_bg = "#e0f7fa"
    border_color = "#00acc1"

    st.markdown(
        f"""
        <div style="border-radius:15px; padding:20px; margin-top:20px;
                    background-color:{card_bg}; border:2px solid {border_color};
                    text-align:center; font-size:22px; font-weight:bold; color:#333;">
            🔎 Prediction Result: <br>
            <span style="font-size:27px; color:{border_color};">{top_label}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Confidence ----
    st.markdown("### Confidence:")
    st.progress(int(top_conf))
    st.markdown(
        f"""
        <div style="font-size:18px; margin-top:8px; text-align:center;">
            <b>{top_conf:.2f}%</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Top 3 Predictions ----
    st.markdown("### 🥇 Top 3 Predictions:")
    top3_idx = probs.argsort()[-3:][::-1]

    for idx in top3_idx:
        st.write(f"- *{class_names[idx]}* — {probs[idx]*100:.2f}%")

    # ---- Probability Graph ----
    st.markdown("### 📊 Probability Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_idx = probs.argsort()[::-1][:7]  
    ax.bar([class_names[i] for i in sorted_idx], probs[sorted_idx] * 100)
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Top Confidence Scores")
    st.pyplot(fig)

# ---- Footer ----
st.markdown(
    """
    <div style="text-align:center; font-size:12px; color:#666; margin-top:30px;">
    ⚠ This tool is for informational purposes only and is <b>not a substitute for professional medical advice</b>.
    </div>
    """,
    unsafe_allow_html=True
)
