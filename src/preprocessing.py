# LIMPIEZA Y PREPARACIÓN DEL DATASET

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: valores nulos, tipos y columnas irrelevantes"""

    print("\n🧹 Iniciando limpieza de datos...")

    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Reemplazar valores problemáticos con NaN
    df.replace(['?', 'None', 'nan', 'NaN', 'null', '-', '—'], np.nan, inplace=True)

    # --- Corrección de tipos numéricos ---
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
    df['Levy'] = df['Levy'].astype(str).str.replace('—', '0').astype(float)
    df['Prod. year'] = df['Prod. year'].astype(int)
    df['Mileage'] = df['Mileage'].astype(str).str.replace(' km', '').astype(float)
    df['Engine volume'] = df['Engine volume'].astype(str).str.replace(' Turbo', '').astype(float)
    df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce')

    # --- Variables binarias ---
    df['Leather interior'] = df['Leather interior'].map({'Yes': 1, 'No': 0})
    df['Wheel'] = df['Wheel'].map({'Left wheel': 0, 'Right-hand drive': 1})

    # --- Completar valores nulos ---
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # --- Eliminar columnas innecesarias ---
    df.drop(columns=['ID', 'Model'], inplace=True, errors='ignore')

    print("✅ Limpieza completa. Dataset listo para preparación.")
    return df

def prepare_data(df: pd.DataFrame):
    """Realiza preprocesamiento básico: convierte texto en números y separa variables"""

    print("\n🔧 Preparando los datos...")

    # Convertir variables categóricas en numéricas
    df = pd.get_dummies(df, drop_first=True)

    # Separar variables (X = características, y = valor a predecir)
    X = df.drop("Price", axis=1)
    y = df["Price"]

    # Dividir en entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Datos preparados correctamente.")
    print(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
    return X_train, X_test, y_train, y_test
