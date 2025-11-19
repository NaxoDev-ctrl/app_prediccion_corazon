import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

st.set_page_config(page_title="Predicciones Médicas", page_icon="🫀")

st.title("🫀 Predicción de Riesgo Cardiovascular")
st.write("Red Neuronal Artificial para evaluación temprana de riesgo cardíaco")

# Verificar que existe el modelo
try:
    model = keras.models.load_model('modulos/corazon_model.h5')
    scaler = joblib.load('modulos/corazon_scaler.pkl')
    features = joblib.load('modulos/corazon_features.pkl')
except:
    st.error("❌ Modelo no encontrado. Primero ejecuta: python modelo_corazon.py")
    st.stop()

# FORMULARIO DE ENTRADA
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Datos Personales")
    age = st.number_input("Edad", 18, 120, 50)
    gender = st.selectbox("Género", ["Hombre", "Mujer"])
    weight = st.number_input("Peso (kg)", 30.0, 200.0, 70.0)
    height = st.number_input("Altura (cm)", 100.0, 250.0, 170.0)
    bmi = weight / ((height/100) ** 2)
    st.info(f"📊 IMC: {bmi:.1f}")

with col2:
    st.subheader("🏃 Estilo de Vida")
    smoking = st.selectbox("¿Fuma?", ["Nunca", "Anteriormente", "Actualmente"])
    alcohol = st.selectbox("Alcohol", ["Ninguno", "Bajo", "Moderado", "Alto"])
    activity = st.selectbox("Actividad Física", ["Sedentario", "Moderado", "Activo"])
    diet = st.selectbox("Dieta", ["Poco saludable", "Promedio", "Saludable"])
    stress = st.selectbox("Estrés", ["Bajo", "Medio", "Alto"])
with col3:
    st.subheader("🩺 Historial Médico")
    hypertension = st.selectbox("Hipertensión", ["No", "Sí"])
    diabetes = st.selectbox("Diabetes", ["No", "Sí"])
    hyperlipidemia = st.selectbox("Hiperlipidemia", ["No", "Sí"])
    family_history = st.selectbox("Historial Familiar", ["No", "Sí"])
    previous_heart = st.selectbox("Infarto Previo", ["No", "Sí"])
    
st.markdown("### 📈 Mediciones Clínicas")
col4, col5, col6, col7 = st.columns(4)
with col4:
    systolic = st.number_input("Presión Sistólica", 80, 200, 120)
with col5:
    diastolic = st.number_input("Presión Diastólica", 50, 130, 80)
with col6:
    heart_rate = st.number_input("Frecuencia Cardíaca", 40, 150, 70)
with col7:
    blood_sugar = st.number_input("Glucosa", 70, 300, 100)

col8, col9 = st.columns(2)
with col8:
    cholesterol = st.number_input("Colesterol Total", 100, 400, 200)
with col9:
    st.write("")  # Espaciador

# BOTÓN DE PREDICCIÓN
st.markdown("---")
if st.button("🔍 PREDECIR RIESGO", type="primary", use_container_width=True):
    
    # Preparar datos (exactamente como en el dataset)
    input_data = {
        'Age': age,
        'Weight': weight,
        'Height': height,
        'BMI': bmi,
        'Hypertension': 1 if hypertension == "Sí" else 0,
        'Diabetes': 1 if diabetes == "Sí" else 0,
        'Hyperlipidemia': 1 if hyperlipidemia == "Sí" else 0,
        'Family_History': 1 if family_history == "Sí" else 0,
        'Previous_Heart_Attack': 1 if previous_heart == "Sí" else 0,
        'Systolic_BP': systolic,
        'Diastolic_BP': diastolic,
        'Heart_Rate': heart_rate,
        'Blood_Sugar_Fasting': blood_sugar,
        'Cholesterol_Total': cholesterol
    }
    
    # One-hot encoding para variables categóricas
    # Gender
    input_data['Gender_Male'] = 1 if gender == "Male" else 0
    
    # Smoking
    for level in ['Current', 'Former', 'Never']:
        input_data[f'Smoking_{level}'] = 1 if smoking == level else 0
    
    # Alcohol_Intake
    for level in ['High', 'Low', 'Moderate', 'None']:
        input_data[f'Alcohol_Intake_{level}'] = 1 if alcohol == level else 0
    
    # Physical_Activity
    for level in ['Active', 'Moderate', 'Sedentary']:
        input_data[f'Physical_Activity_{level}'] = 1 if activity == level else 0
    
    # Diet
    for level in ['Average', 'Healthy', 'Unhealthy']:
        input_data[f'Diet_{level}'] = 1 if diet == level else 0
    
    # Stress_Level
    for level in ['High', 'Low', 'Medium']:
        input_data[f'Stress_Level_{level}'] = 1 if stress == level else 0
    
    # Crear DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Asegurar mismas columnas que entrenamiento
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[features]
    
    # Normalizar y predecir
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled, verbose=0)[0][0]
    
    # MOSTRAR RESULTADOS
    st.markdown("---")
    st.subheader("📊 Resultado de la Predicción")
    
    # Barra de progreso
    st.progress(float(prediction))
    st.metric("Probabilidad de Riesgo", f"{prediction*100:.1f}%")
    
    # Interpretación
    if prediction < 0.3:
        st.success("✅ **RIESGO BAJO** - Mantener hábitos saludables y controles anuales")
    elif prediction < 0.7:
        st.warning("⚠️ **RIESGO MEDIO** - Evaluación médica recomendada + exámenes complementarios")
    else:
        st.error("🚨 **RIESGO ALTO** - Consultar urgentemente con cardiólogo")
    
    # Información adicional
    with st.expander("ℹ️ Información del modelo"):
        st.write("""
        **Características del modelo:**
        - Arquitectura: Red Neuronal MLP (64→32→16→1)
        - Dataset: Synthetic Heart Disease Prediction (Kaggle)
        - Variables: 20 características (demográficas, estilo de vida, historial médico, mediciones clínicas)
        
        **Variables consideradas:**
        - Demográficas: edad, género, peso, altura, IMC
        - Estilo de vida: tabaquismo, alcohol, actividad física, dieta, estrés
        - Historial: hipertensión, diabetes, hiperlipidemia, historial familiar, infarto previo
        - Mediciones: presión arterial, frecuencia cardíaca, glucosa, colesterol
        
        ⚠️ **Disclaimer:** Herramienta de apoyo. NO reemplaza diagnóstico médico profesional.
        """)
