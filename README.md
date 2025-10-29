# 🧠 Algoritmo Predictivo – Probabilidad y Estadística 2025

Proyecto de análisis y predicción de datos con Python.
Incluye scripts para explorar, limpiar datos y entrenar modelos predictivos.

---

## 🚀 Instalación y ejecución

Una vez clonado el repositorio:

### 1. Descomprimir el dataset

1. Descomprimir el dataset (`car-price-prediction-dataset.zip`) en el directorio raíz del proyecto.
2. Mover el archivo `car_price_prediction.csv` al directorio `data/`.

### 2. Crear el entorno virtual

```bash
# En Linux/Mac:
python3 -m venv venv
source venv/bin/activate

# En Windows:
python -m venv venv
venv\Scripts\activate
```

Si ves algo como `(venv)` al inicio del prompt, el entorno está activo.

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

Si el archivo requirements.txt no existe aún, podés crearlo ejecutando:

```bash
pip freeze > requirements.txt
```

### 4. Ejecutar el script principal

```bash
python ./src/main.py
```

### 5. Detener el entorno virtual

```bash
deactivate
```

## Estructura del proyecto

```bash
├── data
│   └── car_price_prediction.csv
├── src
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── visualization.py
├── main.py
├── README.md
└── requirements.txt
```

- `data_loader.py`: Carga el dataset desde un archivo CSV.
- `preprocessing.py`: Realiza preprocesamiento y limpieza de los datos.
- `models.py`: Entrena modelos predictivos (regresión lineal y árbol de decisión).
- `evaluation.py`: Evalúa los modelos y compara sus desempeños.
- `visualization.py`: Genera gráficos y visualizaciones para analizar los datos.
