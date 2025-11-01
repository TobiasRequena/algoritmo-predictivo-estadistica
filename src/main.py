# PUNTO DE ENTRADA DEL PROYECTO

from data_loader import load_data
from preprocessing import clean_data, prepare_data
from models import train_linear_model, train_tree_model
from evaluation import compare_models


def main():
    print("🚗 Iniciando proyecto de predicción de precios de autos...\n")

    # 1. Cargar dataset
    df = load_data("data/car_price_prediction.csv")

    # 2. Preparar datos
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data(df_clean)

    # 3. Entrenar modelos
    # model_lr, metrics_lr = train_linear_model(X_train, X_test, y_train, y_test)
    # model_tree, metrics_tree = train_tree_model(X_train, X_test, y_train, y_test)

    # 4. Evaluar resultados
    # compare_models(metrics_lr, metrics_tree)

    print("\n✅ Proceso finalizado correctamente.")

if __name__ == "__main__":
    main()
