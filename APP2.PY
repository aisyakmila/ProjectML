import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import pickle

# Load Model dan LabelEncoder
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("final_model.h5")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# Fungsi Preprocessing (hanya RGB 3 channel)
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    img_rgb = np.array(image)
    img_rgb = img_rgb.astype("float32") / 255.0
    img_rgb = np.expand_dims(img_rgb, axis=0)
    return img_rgb

# === Menu Sidebar ===
menu = ["HOME", "CHECK YOUR UNDERTONE HERE"]
choice = st.sidebar.selectbox("Navigasi", menu)

# === HOME ===
if choice == "HOME":
    st.title("Welcome to Undertone Finder💅🏻")
    st.markdown("""
    ## Apa itu Undertone?  
    Undertone adalah warna dasar alami kulit yang tidak berubah meskipun warna kulitmu berubah karena paparan matahari.  
    Mengetahui undertone kulitmu bisa membantu kamu memilih warna pakaian, makeup, dan aksesori yang paling cocok.

    ### Jenis Undertone:
    - **Cool** - Nada kebiruan atau merah muda
    - **Warm** - Nada kekuningan atau keemasan
    - **Neutral** - Campuran antara cool dan warm
    """)
                
    st.image("undertone.png", use_container_width=True)

    st.markdown("""
    ### Apa yang Bisa Kamu Lakukan di Aplikasi Ini?
    - 🔍 Deteksi undertone dari gambar nadi (upload gambar)
    - 📷 Deteksi realtime dari kamera
    - 🎨 Dapatkan rekomendasi warna yang sesuai dengan undertonemu

    ---
    Yuk mulai deteksi di menu sebelah! 👈🏻👈🏻👈🏻
    """)

# === DETEKSI UNDERTONE ===
elif choice == "CHECK YOUR UNDERTONE HERE":
    st.title("🔍 Deteksi Undertone Kulit")
    st.write("Pilih metode input gambar:")

    tab1, tab2 = st.tabs(["📁 Upload File", "📷 Kamera Realtime"])

    with tab1:
        uploaded_file = st.file_uploader("Upload gambar nadi (jpg, png)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diupload", use_container_width=True)

            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            predicted_index = np.argmax(prediction)
            predicted_class = label_encoder.inverse_transform([predicted_index])[0]
            confidence = np.max(prediction)

            st.markdown("### Hasil Prediksi")
            st.success(f"Undertone kamu: **{predicted_class}**")
            st.info(f"Tingkat keyakinan model: **{confidence*100:.2f}%**")

            st.markdown("### Rekomendasi Warna Outfit 👗")
            if predicted_class == "Cool":
                st.write("- Warna cocok: Biru, Ungu, Abu-abu, Silver")
                st.image("COOL.png", caption="Palet Warna untuk Undertone Cool", width=300)
            elif predicted_class == "Warm":
                st.write("- Warna cocok: Kuning, Coklat, Emas, Hijau Zaitun")
                st.image("WARM.png", caption="Palet Warna untuk Undertone Warm", width=300)
            else:
                st.write("- Warna cocok: Beige, Peach, Merah Muda, Mint")
                st.image("NEUTRAL.png", caption="Palet Warna untuk Undertone Neutral", width=300)

    with tab2:
        st.write("Ambil gambar dari kamera (gunakan tombol di bawah)")

        camera_image = st.camera_input("Ambil Gambar dari Kamera")

        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")

            st.image(image, caption="Gambar dari Kamera", use_container_width=True)

            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            predicted_index = np.argmax(prediction)
            predicted_class = label_encoder.inverse_transform([predicted_index])[0]
            confidence = np.max(prediction)

            st.markdown("### Hasil Prediksi")
            st.success(f"Undertone kamu: **{predicted_class}**")
            st.info(f"Tingkat keyakinan model: **{confidence*100:.2f}%**")

            st.markdown("### Rekomendasi Warna Outfit 👗")
            if predicted_class == "Cool":
                st.write("- Warna cocok: Biru, Ungu, Abu-abu, Silver")
                st.image("COOL.png", caption="Palet Warna untuk Undertone Cool", width=300)
            elif predicted_class == "Warm":
                st.write("- Warna cocok: Kuning, Coklat, Emas, Hijau Zaitun")
                st.image("WARM.png", caption="Palet Warna untuk Undertone Warm", width=300)
            else:
                st.write("- Warna cocok: Beige, Peach, Merah Muda, Mint")
                st.image("NEUTRAL.png", caption="Palet Warna untuk Undertone Neutral", width=300)
