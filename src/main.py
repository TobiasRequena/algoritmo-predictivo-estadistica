# PUNTO DE ENTRADA DEL PROYECTO

from data_loader import load_data
from preprocessing import clean_data, prepare_data
from models import train_all_models, save_results_summary
from evaluation import compare_models
from visualization import plot_results, plot_feature_importance


def main():
    # 1. Cargar dataset
    df = load_data("data/car_price_prediction.csv")

    # 2. Preparar datos
    df_clean = clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data(df_clean)

    # 3. Entrenar modelos
    results = train_all_models(X_train, X_test, y_train, y_test)
    save_results_summary(results)

    print("\n📊 Comparación de modelos:")
    for name, m in results.items():
        print(f" - {name:<20} R²(log): {m['r2_log']:.3f} | R²(real): {m['r2_real']:.3f} | MAE(USD): {m['mae_real']:.2f}")

    # 4. Evaluar y graficar resultados
    plot_results()
    plot_feature_importance("models/decision_tree.pkl", X_train.columns)

    print("\n✅ Proceso finalizado correctamente.")

if __name__ == "__main__":
    main()
