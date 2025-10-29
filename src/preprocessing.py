# LIMPIEZA Y PREPARACIÓN DEL DATASET

import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df: pd.DataFrame):
    """Realiza preprocesamiento básico: convierte texto en números y separa variables"""

    print("\n🧹 Preparando los datos...")

    # Convertir variables categóricas en numéricas
    df = pd.get_dummies(df, drop_first=True)

    # Separar variables (X = características, y = valor a predecir)
    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    # Dividir en entrenamiento y prueba (80% - 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("✅ Datos preparados correctamente.")
    print(f"Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
    return X_train, X_test, y_train, y_test
