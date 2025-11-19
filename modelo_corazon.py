import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

# Crear carpeta modulos si no existe
os.makedirs('modulos', exist_ok=True)

print("="*60)
print("🫀 ENTRENAMIENTO - RED NEURONAL ENFERMEDAD CARDÍACA")
print("="*60)

# 1. CARGAR DATOS
print("\n[1/6] Cargando datos...")
df = pd.read_csv('data/synthetic_heart_disease_dataset.csv')
print(f"✓ Dataset cargado: {df.shape}")

# 2. IDENTIFICAR COLUMNA OBJETIVO
print("\n[2/6] Identificando columna objetivo...")
target_col = 'Heart_Disease'  # Columna objetivo del dataset

if target_col not in df.columns:
    print(f"❌ Error: No se encontró la columna '{target_col}'")
    print(f"Columnas disponibles: {df.columns.tolist()}")
    exit()

print(f"✓ Columna objetivo: '{target_col}'")
print(f"  Distribución: {df[target_col].value_counts().to_dict()}")

# 3. PREPROCESAR
print("\n[3/6] Preprocesando datos...")
X = df.drop(target_col, axis=1)
y = df[target_col]

# One-hot encoding para variables categóricas
X = pd.get_dummies(X, drop_first=True)
print(f"✓ Características finales: {X.shape[1]}")

# Dividir datos: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Datos normalizados")

# 4. CREAR MODELO
print("\n[4/6] Construyendo red neuronal...")
model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
print("✓ Arquitectura: Input → 64 → 32 → 16 → 1 (Sigmoid)")

# 5. ENTRENAR
print("\n[5/6] Entrenando modelo...")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

epochs_trained = len(history.history['loss'])
print(f"✓ Entrenamiento completado en {epochs_trained} épocas")

# 6. EVALUAR
print("\n[6/6] Evaluando modelo...")
test_loss, test_accuracy, test_auc = model.evaluate(X_test_scaled, y_test, verbose=0)

y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"\n📊 RESULTADOS:")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  AUC-ROC:  {test_auc:.4f}")

print(f"\n{classification_report(y_test, y_pred, target_names=['Sin Riesgo', 'Con Riesgo'])}")

# 7. GUARDAR
print("\n💾 Guardando modelo...")
model.save('modulos/corazon_model.h5')
joblib.dump(scaler, 'modulos/corazon_scaler.pkl')
joblib.dump(X.columns.tolist(), 'modulos/corazon_features.pkl')

print("\n✅ COMPLETADO")
print("Archivos generados:")
print("  - modulos/corazon_model.h5")
print("  - modulos/corazon_scaler.pkl")
print("  - modulos/corazon_features.pkl")
print("\nAhora ejecuta: streamlit run app.py")
print("="*60)