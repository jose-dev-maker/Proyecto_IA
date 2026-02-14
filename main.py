import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carga de Datos
# Asegúrate de tener el archivo 'diabetes.csv' en la misma carpeta que este script
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('diabetes.csv', sep='|', skiprows=1, names=column_names)

# Limpieza inicial: Eliminar la columna de índice si se cargó accidentalmente
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Convertir todas las columnas a numérico por si hay errores de formato
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# 2. Preprocesamiento de Datos
# En este dataset, los valores 0 en Glucosa, Presión, etc. son datos faltantes, no valores reales.
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Separación de variables predictoras (X) y variable objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Escalado de características para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División del dataset: 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Entrenamiento del Modelo (Aprendizaje Supervisado)
# Usamos Random Forest por su robustez ante datos desbalanceados y su buen rendimiento general
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluación
y_pred = model.predict(X_test)

print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.2f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión visual
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.savefig('matriz_confusion.png')
plt.show()