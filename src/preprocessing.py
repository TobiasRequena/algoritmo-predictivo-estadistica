# LIMPIEZA Y PREPARACIÓN DEL DATASET

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: valores nulos, tipos y columnas irrelevantes"""

    print("\n🧹 INICIANDO LIMPIEZA DE DATOS...")

    df = df.copy()
    initial_shape = df.shape

    # 1. Eliminar duplicados
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"🗑️ Se eliminaron {duplicates} filas duplicadas.")
    df.drop_duplicates(inplace=True)

    # 2. Reemplazar valores problemáticos con NaN
    df.replace(['?', 'None', 'nan', 'NaN', 'null', '-', '—'], np.nan, inplace=True)

    # 3. Contar valores nulos antes
    total_nulls_before = df.isnull().sum().sum()
    print(f"🔍 Valores nulos detectados antes de limpiar: {total_nulls_before}")

    # 4. Corrección de tipos numéricos
    df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
    df['Levy'] = df['Levy'].replace(['—', '-', 'None', 'nan'], 0).astype(float)
    df['Prod. year'] = df['Prod. year'].astype(int)
    df['Mileage'] = df['Mileage'].astype(str).str.replace(' km', '').astype(float)
    df['Engine volume'] = df['Engine volume'].astype(str).str.replace(' Turbo', '').astype(float)
    df['Cylinders'] = pd.to_numeric(df['Cylinders'], errors='coerce')

    # 5. Variables binarias
    df['Leather interior'] = df['Leather interior'].map({'Yes': 1, 'No': 0})
    df['Wheel'] = df['Wheel'].map({'Left wheel': 0, 'Right-hand drive': 1})

    # 6. Completar valores nulos
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    total_nulls_after = df.isnull().sum().sum()
    print(f"✅ Se completaron los valores nulos (antes: {total_nulls_before}, ahora: {total_nulls_after})")

    # 7. Eliminar columnas irrelevantes
    columns_to_drop = ['ID', 'Model']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print(f"🧾 Columnas eliminadas: {columns_to_drop}")

    # 8. Reducir cardinalidad
    def limit_categories(df, column, top_n=10):
        top = df[column].value_counts().index[:top_n]
        df[column] = df[column].where(df[column].isin(top), 'Other')
        return df

    df = limit_categories(df, 'Manufacturer', 15)
    df = limit_categories(df, 'Color', 10)
    df = limit_categories(df, 'Category', 10)

    # 8. Aplicar log-transformación
    df['Price'] = np.log1p(df['Price'])

    # 9. Mostrar resumen final
    print(f"📊 Dimensiones antes: {initial_shape}, después: {df.shape}")
    print("✅ LIMPIEZA COMPLETA")
    return df


def prepare_data(df: pd.DataFrame):
    """Realiza preprocesamiento básico: convierte texto en números y separa variables"""

    print("\n🔧 PREPARANDO LOS DATOS...")

    # 1. Convertir variables categóricas en numéricas
    print("🔄 Aplicando codificación one-hot (get_dummies)...")
    df = pd.get_dummies(df, drop_first=True)
    print(f"📈 Total de columnas después del encoding: {df.shape[1]}")

    # 2. Separar variables
    X = df.drop("Price", axis=1)
    y = df["Price"]

    # 3. Dividir en entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Normalizar solo las columnas numéricas
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    print("✅ DATOS PREPARADOS CORRECTAMENTE")
    print(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
    return X_train, X_test, y_train, y_test
