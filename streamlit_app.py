import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set path relatif ke model
MODEL_PATH = os.path.join('model', 'FixMobileNet30New_model.h5')

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Daftar label kelas
class_names = ['Kaca', 'Kardus', 'Logam', 'Plastik']

# Fungsi prediksi
def predict_image(image):
    img = image.resize((224, 224))  # Sesuaikan input model
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Tambah batch dimension
    img_array = img_array / 255.0  # Normalisasi

    predictions = model.predict(img_array)[0]  # Ambil hasil prediksi
    results = {class_names[i]: float(predictions[i] * 100) for i in range(len(class_names))}
    return results

# Streamlit UI
st.title("🧠 Klasifikasi Gambar Sampah")
st.write("Upload satu atau lebih gambar sampah untuk diklasifikasikan menggunakan model MobileNetV2.")

uploaded_files = st.file_uploader("Pilih gambar sampah", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        st.markdown(f"### 🖼️ Gambar {i+1}")
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Nama File: {uploaded_file.name}", use_column_width=True)

        with st.spinner(f"Memproses gambar {i+1}..."):
            prediction_results = predict_image(image)
            sorted_results = sorted(prediction_results.items(), key=lambda x: x[1], reverse=True)

            st.subheader("📊 Hasil Prediksi:")
            for label, confidence in sorted_results:
                st.write(f"**{label}**: {confidence:.2f}%")

            # Menyoroti hasil prediksi tertinggi
            top_class, top_conf = sorted_results[0]
            st.success(f"Hasil Prediksi Teratas: **{top_class}** ({top_conf:.2f}% yakin)")
        st.markdown("---")  # Garis pemisah antar gambar
