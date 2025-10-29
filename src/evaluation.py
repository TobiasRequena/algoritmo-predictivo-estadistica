# EVALUACIÓN Y COMPARACIÓN DE MODELOS

def compare_models(metrics_lr, metrics_tree):
    """Compara las métricas de ambos modelos"""

    print("\n📊 Comparación de resultados:")
    print(f"Regresión Lineal  → R²: {metrics_lr[0]:.3f}, MAE: {metrics_lr[1]:.3f}")
    print(f"Árbol de Decisión → R²: {metrics_tree[0]:.3f}, MAE: {metrics_tree[1]:.3f}")

    if metrics_tree[0] > metrics_lr[0]:
        print("🏆 El Árbol de Decisión tuvo mejor desempeño general.")
    else:
        print("🏆 La Regresión Lineal tuvo mejor desempeño general.")
