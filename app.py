import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import json


# =====================================================
# CONFIGURACIÓN DE LA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Clasificador de Razas de Mascotas",
    page_icon="🐶",
    layout="centered"
)

# =====================================================
# TÍTULO
# =====================================================

st.title("🐶🐱 Clasificador de Razas de Mascotas")
st.write(
    "Sube una imagen de un perro o gato y el modelo predecirá la raza."
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Información del Proyecto")

st.sidebar.write(
    """
    Esta aplicación utiliza Deep Learning con ResNet50
    para clasificar razas de perros y gatos.

    Dataset utilizado:
    Oxford-IIIT Pet Dataset
    """
)

# =====================================================
# CARGAR NOMBRES DE CLASES
# =====================================================

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# =====================================================
# CARGAR CSV DE ENFERMEDADES
# =====================================================

df_enfermedades = pd.read_csv(
    "enfermedades_por_raza_completo (6).csv",
    sep=";",
    encoding="latin1"
)

# =====================================================
# TRANSFORMACIONES DE IMAGEN
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================================
# CARGAR MODELO
# =====================================================

@st.cache_resource

def load_model():

    num_classes = len(class_names)

    # Cargar ResNet50
    model = models.resnet50(weights=None)

    # MISMA arquitectura usada en entrenamiento
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    # Cargar pesos
    model.load_state_dict(
        torch.load(
            "mejor_resnet.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model

# CARGAR EL MODELO
model = load_model()

# =====================================================
# FUNCIÓN DE PREDICCIÓN
# =====================================================

def predict_image(image):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]

    confidence = confidence.item() * 100

    return predicted_class, confidence, probabilities

# =====================================================
# FUNCIÓN TOP 3 PREDICCIONES
# =====================================================

def get_top_predictions(probabilities, top_k=3):

    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []

    for prob, idx in zip(top_probs[0], top_indices[0]):

        breed = class_names[idx.item()]

        probability = prob.item() * 100

        results.append((breed, probability))

    return results
# =====================================================
# CARGAR IMAGEN
# =====================================================

uploaded_file = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PREDICCIÓN
# =====================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Imagen subida", use_container_width=True)

        with st.spinner("Analizando imagen..."):

            predicted_class, confidence, probabilities = predict_image(image)

            top_predictions = get_top_predictions(probabilities)

        # =============================================
        # RESULTADO PRINCIPAL
        # =============================================

        st.success(f"Raza predicha: {predicted_class}")

        st.info(f"Confianza: {confidence:.2f}%")

        # =============================================
        # TOP 3 PREDICCIONES
        # =============================================

        st.subheader("Top 3 Predicciones")

        top_df = pd.DataFrame(
            top_predictions,
            columns=["Raza", "Probabilidad (%)"]
        )

        st.dataframe(top_df, use_container_width=True)

        # =============================================
        # INFORMACIÓN DE ENFERMEDADES
        # =============================================

        st.subheader("Información de Salud")

        raza_limpia = predicted_class.lower().strip()

        enfermedades = df_enfermedades[
            df_enfermedades["raza"].str.lower().str.strip() == raza_limpia
        ]

        if len(enfermedades) > 0:

    for _, row in enfermedades.iterrows():

        st.write(f"• Enfermedad: {row['enfermedad']}")
        st.write(f"  Gravedad: {row['gravedad']}")
        st.write(f"  Recomendación: {row['recomendacion']}")
        st.write("---")

else:
    st.warning(
        "No se encontró información de enfermedades para esta raza."
    )
# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "Aplicación desarrollada con Streamlit y PyTorch"
)
