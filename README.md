# 🫀 Predicción de Enfermedad Cardíaca con Red Neuronal

Aplicación de Machine Learning para evaluación temprana de riesgo cardiovascular usando Red Neuronal Artificial.

## 🚀 Instalación Rápida

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Descargar dataset
👉 **[Synthetic Heart Disease Prediction Dataset - Kaggle](https://www.kaggle.com/datasets/mragpavank/synthetic-heart-disease-prediction)**

Guardar como: `data/heart_disease.csv`

### 3. Entrenar modelo
```bash
python modelo_corazon.py
```

### 4. Ejecutar aplicación
```bash
streamlit run app.py
```

---

## 📊 Problema

### Contexto
- **17.9 millones** de muertes anuales por enfermedades cardiovasculares (OMS, 2023)
- En Chile: **11,284 muertes** en 2022 (MINSAL)
- **60-70%** de infartos ocurren sin síntomas previos detectables

### Solución Propuesta
Red Neuronal Artificial (MLP) para detección temprana de riesgo cardiovascular que:
- Procesa múltiples factores de riesgo simultáneamente
- Proporciona evaluación objetiva y rápida
- Asiste en triaje y decisiones preventivas

---

## 🧠 Arquitectura del Modelo

```
Entrada (14-18 características)
    ↓
Dense 64 + ReLU + Dropout 30%
    ↓
Dense 32 + ReLU + Dropout 30%
    ↓
Dense 16 + ReLU + Dropout 20%
    ↓
Dense 1 + Sigmoid → Probabilidad [0-1]
```

### Variables de Entrada

**21 características en total:**

- **Demográficas**: Age, Gender, Weight, Height, BMI
- **Estilo de vida**: Smoking, Alcohol_Intake, Physical_Activity, Diet, Stress_Level  
- **Historial médico**: Hypertension, Diabetes, Hyperlipidemia, Family_History, Previous_Heart_Attack
- **Mediciones clínicas**: Systolic_BP, Diastolic_BP, Heart_Rate, Blood_Sugar_Fasting, Cholesterol_Total

### Técnicas Aplicadas
- Dropout (prevención de overfitting)
- Early Stopping (patience=10)
- StandardScaler (normalización)
- One-hot encoding (variables categóricas)

---

## 📈 Métricas de Evaluación

| Métrica | Objetivo |
|---------|----------|
| Accuracy | General |
| AUC-ROC | ≥ 0.88 |
| Sensibilidad | ≥ 0.85 |
| Especificidad | Maximizar |

---

## 🎯 Interpretación de Resultados

| Probabilidad | Nivel | Recomendación |
|--------------|-------|---------------|
| < 30% | **BAJO** | Controles rutinarios anuales |
| 30-70% | **MEDIO** | Evaluación médica + exámenes |
| > 70% | **ALTO** | Consulta urgente con cardiólogo |

---

## 🧪 Ejemplos de Prueba

### Paciente 1: Riesgo Bajo (Perfil Saludable)
```
Age: 35 | Gender: Male | Weight: 70kg | Height: 175cm | BMI: 22.9
Smoking: Never | Alcohol: None | Activity: Active | Diet: Healthy | Stress: Low
Hypertension: No | Diabetes: No | Hyperlipidemia: No
Family_History: No | Previous_Heart_Attack: No
Systolic_BP: 110 | Diastolic_BP: 70 | Heart_Rate: 70
Blood_Sugar: 90 | Cholesterol: 180
→ Resultado esperado: <30% (RIESGO BAJO)
```

### Paciente 2: Riesgo Alto (Múltiples Factores)
```
Age: 65 | Gender: Male | Weight: 95kg | Height: 170cm | BMI: 32.9
Smoking: Current | Alcohol: High | Activity: Sedentary | Diet: Unhealthy | Stress: High
Hypertension: Sí | Diabetes: Sí | Hyperlipidemia: Sí
Family_History: Sí | Previous_Heart_Attack: Sí
Systolic_BP: 160 | Diastolic_BP: 100 | Heart_Rate: 90
Blood_Sugar: 180 | Cholesterol: 280
→ Resultado esperado: >70% (RIESGO ALTO)
```

---

## 📁 Estructura del Proyecto

```
proyecto/
├── data/
│   └── heart_disease.csv          # Dataset de Kaggle
├── modulos/                        # Generado automáticamente
│   ├── corazon_model.h5           # Red neuronal entrenada
│   ├── corazon_scaler.pkl         # Normalizador
│   └── corazon_features.pkl       # Nombres de características
├── modelo_corazon.py              # Script de entrenamiento
├── app.py                         # Aplicación Streamlit
├── requirements.txt               # Dependencias
└── README.md                      # Documentación
```

---

## 🔮 Mejoras Futuras

### Corto plazo
- SHAP values para interpretabilidad
- Calibración de probabilidades
- Umbrales personalizados por edad

### Mediano plazo
- Modelos ensemble (MLP + RF + XGBoost)
- Series temporales (tendencias)
- Datos regionales de Chile

### Largo plazo
- Integración con IoT/wearables
- Federated Learning entre hospitales
- Aprobación ISP como dispositivo médico

---

## 👥 Integrantes

- Tamara Larenas
- Ivan Hernandez
- Ignacio Sanhueza

---

## ⚠️ Disclaimer

**Esta aplicación es una herramienta de apoyo educativa y NO reemplaza el diagnóstico médico profesional.**

Las predicciones deben ser interpretadas por personal médico calificado.

---

## 📄 Licencia

Proyecto académico - Universidad de Los Lagos