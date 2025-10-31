# ENTRENAMIENTO DE MODELOS PREDICTIVOS

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def train_linear_model(X_train, X_test, y_train, y_test):
    """Entrena un modelo de regresión lineal"""
    print("\n📈 Entrenando modelo: Regresión Lineal...")

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"   → R²: {r2:.3f}, MAE: {mae:.3f}")
    return model, (r2, mae)


def train_tree_model(X_train, X_test, y_train, y_test):
    """Entrena un modelo de árbol de decisión"""
    print("\n🌳 Entrenando modelo: Árbol de Decisión...")

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"   → R²: {r2:.3f}, MAE: {mae:.3f}")

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
    plt.figure(figsize=(16, 10))
    plot_tree(
        model,
        filled=True,
        feature_names=X_train.columns,
        rounded=True,
        fontsize=8
    )
    plt.title("Árbol de Decisión – Predicción de precios de autos")
    plt.show()

    return model, (r2, mae)
