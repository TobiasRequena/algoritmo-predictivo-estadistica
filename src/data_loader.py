# CARGA DEL DATASET

import pandas as pd
import os

def load_data(path):
    """Carga el archivo CSV del dataset"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo en {path}")

    df = pd.read_csv(path)
    print(f"✅ Dataset cargado correctamente ({df.shape[0]} filas, {df.shape[1]} columnas)")
    print("Columnas disponibles:", list(df.columns))
    return df
