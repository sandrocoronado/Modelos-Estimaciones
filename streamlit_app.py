import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

# Función para cargar y preprocesar datos
@st.cache_data
def load_data():
    aprobacion_file_path = 'data/Aprobacion.xlsx'
    aprobacion_data = pd.read_excel(aprobacion_file_path)

    le = LabelEncoder()
    aprobacion_data['Eficiencia Perfil'] = le.fit_transform(aprobacion_data['Eficiencia Perfil'])
    aprobacion_data['Eficiencia Perfil-PO'] = le.fit_transform(aprobacion_data['Eficiencia Perfil-PO'])
    aprobacion_data['Eficiencia PO-Negociacion'] = le.fit_transform(aprobacion_data['Eficiencia PO-Negociacion'])
    aprobacion_data['Eficiencia Aprobacion'] = le.fit_transform(aprobacion_data['Eficiencia Aprobacion'])

    X_aprob_corrected = aprobacion_data[['Monto', 'Eficiencia Perfil', 'Eficiencia Perfil-PO', 'Eficiencia PO-Negociacion', 'Eficiencia Aprobacion']]
    y_aprob_corrected = aprobacion_data['CartaConsulta-Aprobacion']

    Q1_aprob_corrected = X_aprob_corrected.quantile(0.25)
    Q3_aprob_corrected = X_aprob_corrected.quantile(0.75)
    IQR_aprob_corrected = Q3_aprob_corrected - Q1_aprob_corrected

    mask_iqr_aprob_corrected = ~((X_aprob_corrected < (Q1_aprob_corrected - 1.5 * IQR_aprob_corrected)) | (X_aprob_corrected > (Q3_aprob_corrected + 1.5 * IQR_aprob_corrected))).any(axis=1)

    X_filtered_iqr_aprob_corrected = X_aprob_corrected[mask_iqr_aprob_corrected]
    y_filtered_iqr_aprob_corrected = y_aprob_corrected[mask_iqr_aprob_corrected]

    return X_filtered_iqr_aprob_corrected, y_filtered_iqr_aprob_corrected

# Función para entrenar el modelo Ridge Regression
@st.cache_resource
def train_ridge_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ridge_model = Ridge()
    ridge_model.fit(X_train, y_train)
    return ridge_model

# Cargar y preprocesar datos
X_filtered_iqr_aprob_corrected, y_filtered_iqr_aprob_corrected = load_data()

# Entrenar el modelo Ridge Regression
ridge_model = train_ridge_model(X_filtered_iqr_aprob_corrected, y_filtered_iqr_aprob_corrected)

# Título de la app
st.title("Estimación de CartaConsulta hasta la Aprobacion")

# Entradas del usuario para las variables X
monto = st.number_input("Monto en Millones de Dólares", min_value=0.0, value=0.0)
eficiencia_perfil = st.selectbox("Eficiencia de Perfil", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_perfil_po = st.selectbox("Eficiencia de Perfil-PO", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_po_negociacion = st.selectbox("Eficiencia PO-Negociacion", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_aprobacion = st.selectbox("Eficiencia Aprobacion", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")

# Widget para seleccionar la fecha inicial
fecha_inicial = st.date_input("Seleccione la fecha de la Carta Consulta", datetime(2024, 7, 12))

# Convertir las entradas del usuario en un DataFrame
input_data = pd.DataFrame({
    'Monto ': [monto],
    'Eficiencia Perfil': [eficiencia_perfil],
    'Eficiencia Perfil-PO': [eficiencia_perfil_po],
    'Eficiencia PO-Negociacion': [eficiencia_po_negociacion],
    'Eficiencia Aprobacion': [eficiencia_aprobacion]
})

# Botón para realizar las predicciones
if st.button("Enviar"):
    # Hacer predicciones con el modelo Ridge Regression
    ridge_pred = ridge_model.predict(input_data)[0]
    
    # Calcular la fecha final sumando la estimación a la fecha inicial
    fecha_final = fecha_inicial + timedelta(days=ridge_pred * 30)  # asumiendo que la estimación está en meses
    
    # Mostrar la predicción y la fecha final
    st.write(f"Estimación en Meses de Aprobación: {ridge_pred:.2f} meses")
    st.write(f"Fecha final estimada: {fecha_final.strftime('%d/%m/%Y')}")
