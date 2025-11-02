# GRÁFICOS Y VISUALIZACIONES

import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np

def plot_results(summary_path="results/summary.csv"):
    """Genera gráficos comparativos de desempeño de modelos"""
    if not os.path.exists(summary_path):
        print("❌ No se encontró el archivo de resultados.")
        return

    df = pd.read_csv(summary_path)
    print("\n📊 Resultados cargados para visualización:\n", df)

    # ===== Gráfico 1: R² comparativo =====
    plt.figure(figsize=(6, 4))
    plt.bar(df["Modelo"], df["R2 (real)"], color=["#4CAF50", "#2196F3"])
    plt.title("Comparación de R² (escala real)")
    plt.ylabel("R²")
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("results/comparacion_r2.png", bbox_inches="tight")
    plt.close()

    # ===== Gráfico 2: MAE comparativo =====
    plt.figure(figsize=(6, 4))
    plt.bar(df["Modelo"], df["MAE (USD)"], color=["#FFC107", "#FF5722"])
    plt.title("Comparación de MAE (menor es mejor)")
    plt.ylabel("Error Medio Absoluto (USD)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("results/comparacion_mae.png", bbox_inches="tight")
    plt.close()

    print("✅ Gráficos guardados en la carpeta /results")

def plot_feature_importance(model_path="models/decision_tree.pkl", feature_names=None):
    """Grafica las 10 variables más importantes del Árbol de Decisión"""

    model = joblib.load(model_path)

    if feature_names is None:
        print("⚠️ No se proporcionaron los nombres de las columnas. No se puede graficar.")
        return

    # Obtener importancias y ordenar
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # top 10

    plt.figure(figsize=(8, 5))
    plt.barh(np.array(feature_names)[indices][::-1], importances[indices][::-1], color="#009688")
    plt.xlabel("Importancia Relativa")
    plt.title("Importancia de Variables – Árbol de Decisión")
    plt.tight_layout()
    plt.savefig("results/feature_importance_tree.png", bbox_inches="tight")
    plt.close()

    print("✅ Gráfico de importancia de variables guardado en /results/feature_importance_tree.png")
