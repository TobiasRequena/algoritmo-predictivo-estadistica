# ENTRENAMIENTO DE MODELOS PREDICTIVOS

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_graphviz
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
import graphviz

def train_linear_model(X_train, X_test, y_train, y_test):
    """Entrena un modelo de regresión lineal"""
    print("\n📈 Entrenando modelo: Regresión Lineal...")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas en escala logarítmica
    r2_log = r2_score(y_test, y_pred)

    # Reconvertir a escala original (precio real)
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)

    # Métricas en escala real
    r2_real = r2_score(y_test_real, y_pred_real)
    mae_real = mean_absolute_error(y_test_real, y_pred_real)

    print(f"   → R² (log): {r2_log:.3f}, R² (real): {r2_real:.3f}, MAE (USD): {mae_real:,.2f}")
    return model, {
        "r2_log": r2_log,
        "r2_real": r2_real,
        "mae_real": mae_real
    }


def train_tree_model(X_train, X_test, y_train, y_test):
    """Entrena un modelo de árbol de decisión"""
    print("\n🌳 Entrenando modelo: Árbol de Decisión...")

    model = DecisionTreeRegressor(max_depth=16, min_samples_split=20, random_state=63)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas en escala logarítmica
    r2_log = r2_score(y_test, y_pred)

    # Reconvertir a escala original (precio real)
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)

    # Métricas en escala real
    r2_real = r2_score(y_test_real, y_pred_real)
    mae_real = mean_absolute_error(y_test_real, y_pred_real)

    print(f"   → R² (log): {r2_log:.3f}, R² (real): {r2_real:.3f}, MAE (USD): {mae_real:,.2f}")

    # 🔍 Mostrar las características más importantes
    importances = sorted(
        zip(X_train.columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    )
    print("\n🔎 Variables más importantes:")
    for feature, importance in importances[:10]:
        print(f"   {feature:<25} {importance:.3f}")

    # 🌳 Visualización del árbol (solo si el dataset no es muy grande)
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        filled=True,
        feature_names=X_train.columns,
        rounded=True,
        fontsize=10,
        max_depth=3   # 👈 solo muestra los primeros niveles
    )
    plt.title("Árbol de Decisión – Predicción de precios de autos (resumen)")
    plt.savefig("results/decision_tree_summary.png", bbox_inches="tight")
    plt.close()

    dot_data = export_graphviz(
        model,
        out_file=None,
        feature_names=X_train.columns,
        filled=True,
        rounded=True,
        max_depth=16,
        proportion=True
    )

    graph = graphviz.Source(dot_data)
    graph.render("results/decision_tree", format="png", cleanup=True)

    return model, {
        "r2_log": r2_log,
        "r2_real": r2_real,
        "mae_real": mae_real
    }

def save_model(model, name, metrics):
    """Guarda el modelo entrenado y sus métricas"""
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{name}.pkl"
    metrics_path = f"models/{name}_metrics.txt"

    # Guardar modelo entrenado
    joblib.dump(model, model_path)

    # Guardar métricas
    with open(metrics_path, "w") as f:
        f.write(f"R²: {metrics[0]:.3f}\nMAE: {metrics[1]:.3f}\n")

    print(f"💾 Modelo guardado en: {model_path}")
    print(f"📊 Métricas guardadas en: {metrics_path}\n")

def save_results_summary(results, filename="results/summary.csv"):
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame([
        {
            "Modelo": name,
            "R2 (log)": metrics["r2_log"],
            "R2 (real)": metrics["r2_real"],
            "MAE (USD)": metrics["mae_real"]
        }
        for name, metrics in results.items()
    ])
    df.to_csv(filename, index=False)
    print(f"📈 Resultados guardados en {filename}")

def train_all_models(X_train, X_test, y_train, y_test):
    results = {}

    # Regresión Lineal
    linear_model, linear_metrics = train_linear_model(X_train, X_test, y_train, y_test)
    results["Regresión Lineal"] = linear_metrics
    save_model(linear_model, "linear_regression", (linear_metrics["r2_real"], linear_metrics["mae_real"]))

    # Árbol de Decisión
    tree_model, tree_metrics = train_tree_model(X_train, X_test, y_train, y_test)
    results["Árbol de Decisión"] = tree_metrics
    save_model(tree_model, "decision_tree", (tree_metrics["r2_real"], tree_metrics["mae_real"]))

    return results
