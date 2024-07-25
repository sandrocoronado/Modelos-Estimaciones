import streamlit as st
import pandas as pd
import joblib

# Cargar los modelos entrenados
ridge_model = joblib.load('ridge_model.pkl')
linear_model = joblib.load('linear_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')

# Título de la app
st.title("Predicción de CartaConsulta-Aprobacion")

# Entradas del usuario para las variables X
monto = st.number_input("Monto", min_value=0.0, value=0.0)
eficiencia_perfil = st.selectbox("Eficiencia de Perfil", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_perfil_po = st.selectbox("Eficiencia de Perfil-PO", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_po_negociacion = st.selectbox("Eficiencia PO-Negociacion", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_aprobacion = st.selectbox("Eficiencia Aprobacion", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")

# Convertir las entradas del usuario en un DataFrame
input_data = pd.DataFrame({
    'Monto': [monto],
    'Eficiencia Perfil': [eficiencia_perfil],
    'Eficiencia Perfil-PO': [eficiencia_perfil_po],
    'Eficiencia PO-Negociacion': [eficiencia_po_negociacion],
    'Eficiencia Aprobacion': [eficiencia_aprobacion]
})

# Hacer predicciones con los modelos cargados
ridge_pred = ridge_model.predict(input_data)[0]
linear_pred = linear_model.predict(input_data)[0]
random_forest_pred = random_forest_model.predict(input_data)[0]

# Mostrar las predicciones
st.write(f"Predicción con Ridge Regression: {ridge_pred:.2f}")
st.write(f"Predicción con Linear Regression: {linear_pred:.2f}")
st.write(f"Predicción con Random Forest: {random_forest_pred:.2f}")