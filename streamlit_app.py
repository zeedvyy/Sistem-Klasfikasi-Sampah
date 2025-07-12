import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Path ke model
MODEL_PATH = os.path.join('model', 'FixMobileNet30New_model.h5')

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Label kelas
class_names = ['Kaca', 'Kardus', 'Logam', 'Plastik']

# Fungsi prediksi
def predict_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)[0]
    results = {class_names[i]: float(predictions[i] * 100) for i in range(len(class_names))}
    return results

# UI Streamlit
st.title("🧠 Klasifikasi Gambar Sampah")
st.write("Upload gambar sampah untuk diklasifikasikan.")

uploaded_files = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Gambar {i+1}", use_column_width=True)
        with st.spinner("Memproses..."):
            prediction = predict_image(image)
            sorted_pred = sorted(prediction.items(), key=lambda x: x[1], reverse=True)
            st.subheader("📊 Hasil Prediksi:")
            for label, conf in sorted_pred:
                st.write(f"**{label}**: {conf:.2f}%")
            top_class, top_conf = sorted_pred[0]
            st.success(f"Prediksi Teratas: **{top_class}** ({top_conf:.2f}%)")
