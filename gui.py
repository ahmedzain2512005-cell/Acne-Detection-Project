import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

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

    # ---- Prediction with Spinner ----
    with st.spinner("⏳ Analyzing image..."):
        time.sleep(1)
        results = model.predict(image)
    
    # ---- Extract Top Prediction ----
    probs = results[0].probs.data.cpu().numpy()
    top_idx = probs.argmax()
    top_label = class_names[top_idx]
    top_conf = probs[top_idx] * 100

    # ---- Display Result Card ----
    card_bg = "#e0f7fa"       # light cyan background
    border_color = "#00acc1"  # border and title color

    st.markdown(
        f"""
        <div style="border-radius:15px; padding:20px; margin-top:20px;
                    background-color:{card_bg}; border:2px solid {border_color};
                    text-align:center; font-size:20px; font-weight:bold; color:#333;">
            🔎 Prediction Result: <br>
            <span style="font-size:24px; color:{border_color};">{top_label}</span><br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Confidence Progress Bar ----
    st.markdown("**Confidence:**")
    st.progress(int(top_conf))

# ---- Footer Disclaimer ----
st.markdown(
    """
    <div style="text-align:center; font-size:12px; color:#666; margin-top:30px;">
    ⚠️ This tool is for informational purposes only and is <b>not a substitute for professional medical advice</b>.
    </div>
    """,
    unsafe_allow_html=True
)