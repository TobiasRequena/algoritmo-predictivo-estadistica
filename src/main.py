# PUNTO DE ENTRADA DEL PROYECTO

from data_loader import load_data
from preprocessing import clean_data, prepare_data
from models import train_all_models, save_results_summary
from evaluation import compare_models


def main():
    print("🚗 Iniciando proyecto de predicción de precios de autos...\n")

    # 1. Cargar dataset
    df = load_data("data/car_price_prediction.csv")

    # 2. Preparar datos
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data(df_clean)

    # 3. Entrenar modelos
    results = train_all_models(X_train, X_test, y_train, y_test)
    save_results_summary(results)

    print("\n📊 Comparación de modelos:")
    for name, (r2, mae) in results.items():
        print(f" - {name:<20} R²: {r2:.3f} | MAE: {mae:.3f}")

    # 4. Evaluar resultados
    # compare_models(metrics_lr, metrics_tree)

    print("\n✅ Proceso finalizado correctamente.")

if __name__ == "__main__":
    main()
