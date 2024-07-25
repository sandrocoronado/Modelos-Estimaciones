import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Cargar el archivo Excel
aprobacion_file_path = 'data/Aprobacion.xlsx'
aprobacion_data = pd.read_excel(aprobacion_file_path)

# Convertir las variables categóricas a numéricas
le = LabelEncoder()
aprobacion_data['Eficiencia Perfil'] = le.fit_transform(aprobacion_data['Eficiencia Perfil'])
aprobacion_data['Eficiencia Perfil-PO'] = le.fit_transform(aprobacion_data['Eficiencia Perfil-PO'])
aprobacion_data['Eficiencia PO-Negociacion'] = le.fit_transform(aprobacion_data['Eficiencia PO-Negociacion'])
aprobacion_data['Eficiencia Aprobacion'] = le.fit_transform(aprobacion_data['Eficiencia Aprobacion'])

# Preparar las características (X) y el objetivo (y)
X_aprob_corrected = aprobacion_data[['Monto', 'Eficiencia Perfil', 'Eficiencia Perfil-PO', 'Eficiencia PO-Negociacion', 'Eficiencia Aprobacion']]
y_aprob_corrected = aprobacion_data['CartaConsulta-Aprobacion']

# Calcular el IQR para cada característica
Q1_aprob_corrected = X_aprob_corrected.quantile(0.25)
Q3_aprob_corrected = X_aprob_corrected.quantile(0.75)
IQR_aprob_corrected = Q3_aprob_corrected - Q1_aprob_corrected

# Identificar outliers utilizando el método IQR
mask_iqr_aprob_corrected = ~((X_aprob_corrected < (Q1_aprob_corrected - 1.5 * IQR_aprob_corrected)) | (X_aprob_corrected > (Q3_aprob_corrected + 1.5 * IQR_aprob_corrected))).any(axis=1)

# Filtrar el dataset para eliminar outliers
X_filtered_iqr_aprob_corrected = X_aprob_corrected[mask_iqr_aprob_corrected]
y_filtered_iqr_aprob_corrected = y_aprob_corrected[mask_iqr_aprob_corrected]

# Dividir los datos filtrados en conjuntos de entrenamiento y prueba
X_train_filtered_iqr_aprob_corrected, X_test_filtered_iqr_aprob_corrected, y_train_filtered_iqr_aprob_corrected, y_test_filtered_iqr_aprob_corrected = train_test_split(X_filtered_iqr_aprob_corrected, y_filtered_iqr_aprob_corrected, test_size=0.2, random_state=42)

# Inicializar los modelos
ridge_model = Ridge()
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor()

# Entrenar los modelos
ridge_model.fit(X_train_filtered_iqr_aprob_corrected, y_train_filtered_iqr_aprob_corrected)
linear_model.fit(X_train_filtered_iqr_aprob_corrected, y_train_filtered_iqr_aprob_corrected)
random_forest_model.fit(X_train_filtered_iqr_aprob_corrected, y_train_filtered_iqr_aprob_corrected)

# Guardar los modelos entrenados
joblib.dump(ridge_model, 'ridge_model.pkl')
joblib.dump(linear_model, 'linear_model.pkl')
joblib.dump(random_forest_model, 'random_forest_model.pkl')
