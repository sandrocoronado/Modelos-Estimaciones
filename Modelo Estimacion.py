import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, date, timedelta

# Función para cargar y preprocesar datos
@st.cache_data
def load_data():
    aprobacion_file_path = 'data/Aprobacion.xlsx'  # Actualiza la ruta al archivo
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

# Definir la fórmula del modelo de Regresión Polinomial (grado 2)
def polynomial_model(X):
    return 0.1094 + 0.1171 * X + 0.0047 * X**2

# Cargar y preprocesar datos
X_filtered_iqr_aprob_corrected, y_filtered_iqr_aprob_corrected = load_data()

# Entrenar el modelo Ridge Regression
ridge_model = train_ridge_model(X_filtered_iqr_aprob_corrected, y_filtered_iqr_aprob_corrected)

# Título de la app
st.title("Estimación y Simulación de Regresión")

# Sección 1: Estimación de CartaConsulta hasta la Aprobacion
st.header("Estimación de Aprobación hasta la CartaConsulta")

# Entradas del usuario para las variables X
monto = st.number_input("Monto en Millones de Dólares", min_value=0.0, value=0.0)
eficiencia_perfil = st.selectbox("Eficiencia de Perfil", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_perfil_po = st.selectbox("Eficiencia de Perfil-PO", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_po_negociacion = st.selectbox("Eficiencia PO-Negociacion", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")
eficiencia_aprobacion = st.selectbox("Eficiencia Aprobacion", options=[0, 1], format_func=lambda x: "Eficiente" if x == 1 else "Con Demora")

# Widget para seleccionar la fecha inicial
fecha_inicial = st.date_input("Seleccione la fecha de la Carta Consulta", date(2024, 7, 12))

# Convertir las entradas del usuario en un DataFrame
input_data = pd.DataFrame({
    'Monto': [monto],
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

# Sección 2: Simulación de la fórmula de regresión polinomial
st.header("Simulación de la Curva de Desembolsos de la Cartera de Brasil")

input_date = st.date_input('Seleccione una fecha para la simulación', date(2024, 7, 31))
input_datetime = datetime.combine(input_date, datetime.min.time())

# Añadir 6.1 años a la fecha seleccionada
years_to_add = 6.108
result_date = input_datetime + timedelta(days=years_to_add * 365.25)  # Usando 365.25 para considerar años bisiestos

st.write(f'Fecha de Cuando se desembolsará el 100%, es resultante al añadir {years_to_add} años a la fecha inicial: {result_date.strftime("%d/%m/%Y")}')

year = input_datetime.year + (input_datetime - datetime(input_datetime.year, 1, 1)).days / 365

# Generar predicciones en incrementos de 1 año
X_simulated = np.arange(0, int(year) + 1, 1)
Y_simulated = polynomial_model(X_simulated)

# Generar predicciones adicionales en incrementos de 1 años cerca de 1
if Y_simulated[-1] < 1:
    X_fine = np.arange(X_simulated[-1], year + 1, 1)
    Y_fine = polynomial_model(X_fine)
    X_simulated = np.concatenate((X_simulated, X_fine))
    Y_simulated = np.concatenate((Y_simulated, Y_fine))

# Filtrar los valores hasta que el Porcentaje Acumulado alcance 1
mask = Y_simulated <= 1
X_simulated = X_simulated[mask]
Y_simulated = Y_simulated[mask]

# Crear un DataFrame con los valores simulados
simulated_values = pd.DataFrame({'Año': X_simulated, 'Porcentaje Acumulado': Y_simulated})

# Plotting the results with Altair
base = alt.Chart(simulated_values).mark_line(point=True).encode(
    x=alt.X('Año', title='Año'),
    y=alt.Y('Porcentaje Acumulado', title='Porcentaje Acumulado', scale=alt.Scale(domain=[0, 1])),
    tooltip=['Año', 'Porcentaje Acumulado']
).properties(
    title='Simulación de Año vs Porcentaje Acumulado'
).interactive()

text = base.mark_text(
    align='left',
    baseline='middle',
    dx=10,
    dy=-12
).encode(
    text=alt.Text('Porcentaje Acumulado:Q', format='.2f')
)

chart = base + text

# Display the chart
st.altair_chart(chart, use_container_width=True)

# Display the simulated values in a table
st.write(simulated_values)