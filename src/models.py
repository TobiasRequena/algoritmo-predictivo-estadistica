# ENTRENAMIENTO DE MODELOS PREDICTIVOS

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error

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

    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"   → R²: {r2:.3f}, MAE: {mae:.3f}")
    return model, (r2, mae)
