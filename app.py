import streamlit as st
import joblib

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image

st.set_page_config(page_title="Détection Tumeur", layout="centered")
st.title("🧠 Application d'Aide au Diagnostic IRM")
st.write("Téléversez une image IRM pour détecter la présence d'une tumeur cérébrale.")

# تحميل الموديل - بدلنا السمية هنا
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl') # كان model ولّى kmeans

def extract_features(image):
    img = image.resize((128, 128)) # استعملنا PIL باش نصغرو الصورة
    img = img.convert('L') # حولناها رمادي
    img = np.array(img)

    intensity = np.mean(img)
    
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    texture = graycoprops(glcm, 'contrast')[0, 0]
    
    thresh = (img > 127) * 255 # بدلنا cv2.threshold بـ numpy
    surface = np.sum(thresh == 255)
    
    return [intensity, texture, surface]

uploaded_file = st.file_uploader("Choisissez une image IRM...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Image IRM Téléversée', width=300)

    if st.button('Lancer l\'Analyse'):
        with st.spinner('Analyse en cours...'):
            features = np.array(extract_features(image)).reshape(1, -1)
            features_scaled = scaler.transform(features)
            prediction = kmeans.predict(features_scaled)[0] # كان model ولّى kmeans

            st.subheader("Résultat de l'Analyse")
            if prediction == 1:
                st.error("🔴 Résultat : Tumeur Détectée")
            else:
                st.success("🟢 Résultat : Cas Normal - Aucune tumeur détectée")

            st.subheader("Features Extraites")
            col1, col2, col3 = st.columns(3)
            col1.metric("Intensité", f"{features[0][0]:.2f}")
            col2.metric("Texture", f"{features[0][1]:.4f}")
            col3.metric("Surface", f"{features[0][2]:.0f}")
