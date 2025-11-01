# CARGA Y EXPLORACIÓN DEL DATASET

import pandas as pd
import os

def load_data(path):
    """Carga el archivo CSV y muestra información general del dataset"""

    print("🚗 Iniciando proyecto de predicción de precios de autos...\n")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo en {path}")

    # 1. Cargar el dataset
    df = pd.read_csv(path)
    print(f"✅ Dataset cargado correctamente ({df.shape[0]} filas, {df.shape[1]} columnas)")
    print("Columnas disponibles:", list(df.columns))

    # 2. Mostrar resumen general
    print("\n📋 Información general del dataset:")
    print("-" * 80)
    print(df.info())

    # 3. Mostrar valores únicos por columna (solo para columnas categóricas pequeñas)
    print("\n🔠 Valores únicos por columna:")
    for col in df.columns:
        uniques = df[col].unique()
        if df[col].dtype == "object" and len(uniques) <= 10:
            print(f"  • {col}: {list(uniques)}")
        elif df[col].dtype == "object":
            print(f"  • {col}: {len(uniques)} valores únicos")

    print("-" * 80)
    return df
