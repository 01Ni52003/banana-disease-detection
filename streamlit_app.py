import streamlit as st
import pandas as pd
import json, pathlib, numpy as np
import tensorflow as tf
from PIL import Image
import io
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# Page config
st.set_page_config(
    page_title="Banana Leaf Disease Detector",
    page_icon="🍌",
    layout="centered"
)

# Custom CSS for better design
st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg, #FFFDE7, #F1F8E9);}
    .step-box {
        background: #ffffff;
        padding: 1.2em;
        margin: 1em 0;
        border-radius: 12px;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: #e8f5e9;
        border-left: 6px solid #43a047;
        padding: 1em;
        border-radius: 8px;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://em-content.zobj.net/source/microsoft-teams/363/banana_1f34c.png", width=80)
    st.markdown("**🍌 Banana Leaf Disease Detector**")
    st.caption("Developed by Candy 💻✨")
    st.markdown("---")
    st.markdown("Easily detect banana leaf diseases and get bilingual (தமிழ்/English) treatment guidance.")

# 🔹 Home Page Title (Bilingual)
st.markdown("""
    <div style='text-align:center;'>
        <h1 style='color:#2e7d32;'>🍌 Banana Leaf Disease Detection <br> வாழை இலை நோய் கண்டறிதல்</h1>
        <p style='font-size:18px; color:#555;'>
            Upload an image of a banana leaf to detect disease and get treatment guidance <br>
            வாழை இலை படத்தை பதிவேற்றம் செய்து நோயை கண்டறிந்து சிகிச்சை வழிகாட்டுதலைப் பெறுங்கள்
        </p>
    </div>
    <hr>
""", unsafe_allow_html=True)

# Language selector
language = st.selectbox("Select Language / மொழியைத் தேர்ந்தெடுங்கள்", ["English", "Tamil"])

# Paths
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "banana_effnet.tflite"
CLASS_MAP_PATH = BASE_DIR / "models" / "class_map.json"
TREATMENT_CSV_PATH = BASE_DIR / "treatment" / "treatment_data.csv"

# Load class map
with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
    class_map = json.load(f)
class_names = [class_map[str(i)] for i in range(len(class_map))]

# Load treatment data
treatment_df = pd.read_csv(TREATMENT_CSV_PATH, encoding="utf-8-sig")

# Load model
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction
def predict(img):
    img = Image.open(img).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return class_names[pred_class], confidence

# Upload
uploaded_file = st.file_uploader("📷 Upload a banana leaf image / வாழை இலை படத்தை பதிவேற்றவும்", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="🖼️ Uploaded Image / பதிவேற்றிய படம்", use_container_width=True)
    label, conf = predict(uploaded_file)

    # Prediction result bilingual
    if language == "English":
        st.markdown(f"<div class='prediction-box'>✅ Prediction: <b>{label}</b><br>🔹 Confidence: {conf:.2%}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-box'>✅ கண்டறிதல்: <b>{label}</b><br>🔹 நம்பிக்கை: {conf:.2%}</div>", unsafe_allow_html=True)

    # Treatment section
    disease_data = treatment_df[treatment_df["Disease"] == label]
    if language == "English":
        st.subheader("📋 Treatment Steps")
    else:
        st.subheader("📋 சிகிச்சை படிகள்")

    total_steps = len(disease_data["Day/Week"].unique())
    done_steps = 0

    # ✅ Fixed grouping to maintain design consistency
    for i, (day, day_df) in enumerate(disease_data.groupby("Day/Week"), start=1):
        with st.container():
            if language == "English":
                st.markdown(f"<div class='step-box'><b>Step {i}</b> ⏰ {day}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='step-box'><b>படி {i}</b> ⏰ {day}</div>", unsafe_allow_html=True)

            # Treatment types
            t_types = list(day_df["Treatment Type (English)"].unique())
            display_map = {"Chemical": "⚗️ Chemical", "Traditional": "🌿 Traditional", "Bio-pesticide": "🦠 Bio-pesticide"}
            t_display = [display_map.get(t, t) for t in t_types]

            selected = st.radio(
                f"Select treatment type for {day}:" if language == "English" else f"{day}க்கு சிகிச்சை வகையைத் தேர்ந்தெடுக்கவும்:",
                t_display,
                horizontal=True,
                index=None,
                key=f"radio_{i}"
            )

            if selected:
                selected_type = t_types[t_display.index(selected)]
                row = day_df[day_df["Treatment Type (English)"] == selected_type].iloc[0]
                if language == "English":
                    st.write(f"**Item:** {row['Product Name (English)']}")
                    st.write(f"**Action:** {row['Action (English)']}")
                else:
                    st.write(f"**பொருள்:** {row['Product Name (Tamil)']}")
                    st.write(f"**செயல்:** {row['Action (Tamil)']}")

                # Buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Accept Treatment" if language == "English" else "👍 சிகிச்சையை ஏற்கவும்", key=f"accept_{i}"):
                        st.success("You accepted this treatment ✅" if language == "English" else "நீங்கள் இந்த சிகிச்சையை ஏற்றுக்கொண்டீர்கள் ✅")
                with col2:
                    if st.checkbox("✔️ Mark done" if language == "English" else "✔️ முடிந்ததாக குறிக்கவும்", key=f"done_{i}"):
                        st.info("This step marked as done" if language == "English" else "இந்த படி முடிந்ததாக குறிக்கப்பட்டது")
                        done_steps += 1

    # ✅ PDF Download (Table format)
    if st.button("📄 Download PDF Report"):
        table_data = [["Step", "Day/Week", "Treatment Type", "Product", "Action"]]

        step_no = 1
        for day, day_df in disease_data.groupby("Day/Week"):
            for _, row in day_df.iterrows():
                table_data.append([
                    f"Step {step_no}",
                    day,
                    row["Treatment Type (English)"],
                    row["Product Name (English)"],
                    row["Action (English)"]
                ])
            step_no += 1

        pdf_buffer = io.BytesIO()
        report = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        table = Table(table_data, colWidths=[60, 80, 110, 90, 180])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        report.build([table])
        pdf_buffer.seek(0)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name="banana_report.pdf",
            mime="application/pdf"
        )

    # Progress bar
    progress = (done_steps / total_steps) if total_steps > 0 else 0
    st.progress(progress)

else:
    if language == "English":
        st.info("Upload an image above to start detection.")
    else:
        st.info("மேலே படத்தை பதிவேற்றம் செய்து கண்டறிதலைத் தொடங்கவும்.")
