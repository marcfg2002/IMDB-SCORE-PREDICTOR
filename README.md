# Estimador de nota IMDb (Mflix)

Aplicación educativa con **Árbol de Decisión**, **Random Forest** y un **sistema experto de reglas**. Usa el dataset tabular Mflix del archivo `archive (1).zip` para estimar la nota IMDb de una película a partir de sus metadatos.

## Qué predice

- **Objetivo:** `imdb.rating`, una variable numérica de 0 a 10; por eso se entrena como un problema de regresión.
- **Entradas:** año, duración, número de votos IMDb, premios ganados, nominaciones y género principal.
- Se usan únicamente películas (`type=movie`) con todos esos datos y una nota disponible. Las series y las filas incompletas se excluyen.
- El entrenamiento lee `mflix.movies.csv` directamente desde el ZIP; no hace falta descomprimir los más de 70 MB de datos.

## Modelos y teoría

- El árbol usa criterio de error cuadrático y profundidad limitada para reducir sobreajuste.
- El bosque entrena árboles con **bootstrap** y subconjuntos de variables. En regresión, el resultado es la **media aritmética** de las estimaciones individuales.
- La evaluación usa un conjunto de prueba reservado: **MAE** (menor es mejor) y **R²** (más próximo a 1 es mejor). El bosque también informa el **R² OOB**.
- El sistema experto mantiene una base de reglas `IF/THEN` legible que interpreta la estimación y los premios. Sus umbrales son didácticos y no alteran el resultado del modelo.

Su arquitectura se presenta como **hechos** (nota estimada y datos de premios), **base de conocimiento** (reglas R1–R7), **motor de inferencia** (condiciones IF/THEN) y **conclusión** (recomendación y reglas activadas).

La estimación es exploratoria y presupone que los votos y premios ya se conocen; no está diseñada para predecir películas futuras sin esos datos.

## Instalación (Windows PowerShell)

```powershell
cd "C:\ruta\al\proyecto"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Entrenamiento

Indica la ruta al ZIP que has descargado:

```powershell
python train.py --dataset-zip "C:\Users\lemvo\Downloads\archive (1).zip"
```

Por defecto se lee `mflix.movies.csv`. Para usar el CSV alternativo de películas embebidas:

```powershell
python train.py --dataset-zip "C:\ruta\archive (1).zip" --csv-name "mflix.embedded_movies.csv"
```

El entrenamiento crea o actualiza `modelo.joblib` en la raíz del proyecto e imprime MAE, R² y OOB R².

## Ejecución

```powershell
python -m streamlit run app.py
```

La aplicación permite introducir los metadatos de una película, comparar las estimaciones del árbol y el bosque, inspeccionar la media y dispersión de las predicciones de los árboles y ver las reglas expertas activadas.

## Estructura

```text
app.py             Interfaz Streamlit y comparación de predicciones
train.py           Lectura de Mflix, entrenamiento y evaluación
expert_system.py   Base de reglas IF/THEN del sistema experto
modelo.joblib      Artefacto generado por train.py
requirements.txt   Dependencias
```

## Deploy en Render

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

Render no entrena automáticamente el modelo. Entrena localmente y publica el `modelo.joblib` generado junto con la aplicación.
