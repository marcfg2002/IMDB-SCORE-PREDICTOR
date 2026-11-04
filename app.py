"""Aplicación educativa de estimación de notas IMDb para películas Mflix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st

from expert_system import evaluate_movie

MODEL_PATH = Path(__file__).resolve().with_name("modelo.joblib")
FEATURE_LABELS = {
    "year": "Año",
    "runtime": "Duración (minutos)",
    "imdb_votes": "Votos IMDb",
    "awards_wins": "Premios ganados",
    "awards_nominations": "Nominaciones",
}


@st.cache_resource
def load_artifact(model_path: Path) -> dict[str, Any]:
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") not in (2, 3):
        raise ValueError(
            "El artefacto no corresponde al modelo Mflix. Vuelve a generarlo con `python train.py`."
        )
    required_keys = {"decision_tree", "random_forest", "metadata"}
    if artifact["artifact_version"] == 3:
        required_keys.add("neural_network")
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise KeyError(f"Faltan claves requeridas en el modelo: {sorted(missing_keys)}")
    return artifact


def bounded_default(metadata: dict[str, Any], feature: str) -> int:
    limits = metadata["feature_ranges"][feature]
    default = metadata["feature_defaults"][feature]
    return max(int(limits["min"]), min(int(round(default)), int(limits["max"])))


def make_feature_input(
    metadata: dict[str, Any], feature: str, label: str
) -> int:
    limits = metadata["feature_ranges"][feature]
    minimum = int(limits["min"])
    maximum = int(limits["max"])
    if minimum == maximum:
        return minimum
    return int(
        st.number_input(
            label,
            min_value=minimum,
            max_value=maximum,
            value=bounded_default(metadata, feature),
            step=1,
        )
    )


st.set_page_config(page_title="Predicción IMDb Mflix", page_icon="🎬", layout="wide")
st.title("🎬 Estimador de nota IMDb para películas")

if not MODEL_PATH.is_file():
    st.error(
        "No se encontró `modelo.joblib`. Entrénalo con "
        "`python train.py --dataset-zip \"C:\\ruta\\archive (1).zip\"`."
    )
    st.stop()

try:
    artifact = load_artifact(MODEL_PATH)
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"No se pudo cargar el modelo: {error}")
    st.stop()

metadata = artifact["metadata"]
metrics = metadata["metrics"]
if "neural_network" in artifact:
    st.caption(
        "Árbol de Decisión, Random Forest y una red neuronal MLP entrenados con metadatos de Mflix; "
        "incluye una prueba de robustez con películas de pocos votos y un sistema experto didáctico."
    )
else:
    st.caption(
        "Árbol de Decisión y Random Forest entrenados con metadatos de Mflix. "
        "Vuelve a entrenar el modelo para habilitar la red neuronal y su prueba de robustez."
    )

with st.expander("📘 Qué hace el proyecto", expanded=False):
    st.markdown(
        """
        - **Objetivo:** estimar la nota IMDb (escala 0–10), no clasificar géneros.
        - **Entradas:** año, duración, votos IMDb, premios, nominaciones y género principal.
        - **Árbol de Decisión:** una secuencia de divisiones; limita la profundidad para reducir sobreajuste.
        - **Random Forest:** árboles entrenados con bootstrap y subconjuntos de variables; promedia sus estimaciones.
        - **Red neuronal MLP:** capas densas con ReLU y Adam; estandariza las entradas y usa parada temprana si hay suficientes muestras.
        - **Prueba de pocos votos:** mide por separado el MAE de la red en el cuartil inferior de votos IMDb.
        - **OOB:** R² calculado con observaciones fuera de bolsa, como comprobación interna del bosque.
        - **Sistema experto:** reglas IF/THEN legibles que interpretan la nota y las señales de premios.
          Es independiente del modelo y no cambia la estimación.

        El sistema experto es una demostración de clase, no una valoración oficial de películas.
        """
    )

st.subheader("Rendimiento en datos no usados para entrenar")
metric_columns = st.columns(3)
metric_columns[0].metric("MAE — Árbol", f"{metrics['decision_tree_mae']:.2f} puntos")
metric_columns[1].metric("MAE — Random Forest", f"{metrics['random_forest_mae']:.2f} puntos")
metric_columns[2].metric("R² OOB — Random Forest", f"{metrics['random_forest_oob_r2']:.3f}")
st.caption(
    f"Prueba: {metrics['test_rows']:,} películas. "
    "MAE más bajo es mejor; R² más cercano a 1 indica mejor ajuste."
)
if "neural_network_mae" in metrics:
    neural_columns = st.columns(2)
    neural_columns[0].metric("MAE — Red neuronal", f"{metrics['neural_network_mae']:.2f} puntos")
    neural_columns[1].metric("R² — Red neuronal", f"{metrics['neural_network_r2']:.3f}")

    st.markdown("#### Prueba de robustez: películas con pocos votos")
    stress_columns = st.columns(4)
    stress_columns[0].metric(
        "MAE general de la red", f"{metrics['neural_network_mae']:.2f} puntos"
    )
    stress_columns[1].metric("Umbral de pocos votos", f"≤ {metrics['low_votes_threshold']:,.0f}")
    stress_columns[2].metric("Películas en el caso especial", metrics["low_votes_test_rows"])
    low_votes_mae = metrics["neural_network_low_votes_mae"]
    stress_columns[3].metric(
        "MAE con pocos votos",
        "Sin casos" if low_votes_mae is None else f"{low_votes_mae:.2f} puntos",
    )
    st.caption(
        "El umbral se calcula con el cuartil inferior de votos del conjunto de entrenamiento; "
        "el error se mide en las películas de prueba que quedan por debajo. "
        "Un MAE mayor que el general indica que este caso es más difícil para la red."
    )

st.sidebar.header("⚙️ Datos de la película")
with st.form("movie-input"):
    genre_column, year_column = st.columns(2)
    with genre_column:
        genre = st.selectbox(
            "Género principal",
            options=metadata["genres"],
            index=metadata["genres"].index(metadata["default_genre"]),
        )
    with year_column:
        year = make_feature_input(metadata, "year", FEATURE_LABELS["year"])

    first_row = st.columns(3)
    with first_row[0]:
        runtime = make_feature_input(metadata, "runtime", FEATURE_LABELS["runtime"])
    with first_row[1]:
        imdb_votes = make_feature_input(metadata, "imdb_votes", FEATURE_LABELS["imdb_votes"])
    with first_row[2]:
        award_wins = make_feature_input(metadata, "awards_wins", FEATURE_LABELS["awards_wins"])
    award_nominations = make_feature_input(
        metadata, "awards_nominations", FEATURE_LABELS["awards_nominations"]
    )
    if "neural_network" in artifact:
        model_options = (
            "Comparar todos",
            "Solo Árbol de Decisión",
            "Solo Random Forest",
            "Solo Red neuronal",
        )
    else:
        model_options = ("Comparar ambos", "Solo Árbol de Decisión", "Solo Random Forest")
    model_mode = st.radio("Modelo que quieres consultar", model_options, horizontal=True)
    submitted = st.form_submit_button("Estimar nota IMDb", type="primary")

if submitted:
    feature_input = {
        "year": year,
        "runtime": runtime,
        "imdb_votes": imdb_votes,
        "awards_wins": award_wins,
        "awards_nominations": award_nominations,
        "genre": genre,
    }

    tree_model = artifact["decision_tree"]
    forest_model = artifact["random_forest"]
    predictions: dict[str, float] = {}
    if model_mode in ("Comparar ambos", "Comparar todos", "Solo Árbol de Decisión"):
        predictions["Árbol de Decisión"] = float(tree_model.predict([feature_input])[0])
    if model_mode in ("Comparar ambos", "Comparar todos", "Solo Random Forest"):
        predictions["Random Forest"] = float(forest_model.predict([feature_input])[0])
    if model_mode in ("Comparar todos", "Solo Red neuronal"):
        neural_prediction = float(artifact["neural_network"].predict([feature_input])[0])
        predictions["Red neuronal"] = float(np.clip(neural_prediction, 0, 10))

    st.subheader("Estimación")
    prediction_columns = st.columns(len(predictions))
    for column, (model_name, prediction) in zip(prediction_columns, predictions.items()):
        column.metric(model_name, f"{prediction:.2f} / 10")

    chosen_prediction = predictions.get("Random Forest", next(iter(predictions.values())))
    if "Random Forest" in predictions:
        vectorizer = forest_model.named_steps["vectorizer"]
        forest = forest_model.named_steps["regressor"]
        encoded_input = vectorizer.transform([feature_input])
        tree_estimates = np.asarray(
            [estimator.predict(encoded_input)[0] for estimator in forest.estimators_],
            dtype=np.float64,
        )
        mean_estimate = float(tree_estimates.mean())
        st.markdown("#### 🌲 Cómo se forma la estimación del bosque")
        forest_columns = st.columns(3)
        forest_columns[0].metric("Media de los árboles", f"{mean_estimate:.2f} / 10")
        forest_columns[1].metric("Estimación mínima", f"{tree_estimates.min():.2f}")
        forest_columns[2].metric("Estimación máxima", f"{tree_estimates.max():.2f}")
        st.caption(
            f"Se promedian {len(tree_estimates)} árboles; "
            "la dispersión muestra cuánto varían sus estimaciones."
        )
        edges = np.histogram_bin_edges(tree_estimates, bins=8)
        counts, edges = np.histogram(tree_estimates, bins=edges)
        chart = {
            "Intervalo de nota": [
                f"{edges[index]:.1f}–{edges[index + 1]:.1f}"
                for index in range(len(counts))
            ],
            "Árboles": counts.tolist(),
        }
        st.bar_chart(chart, x="Intervalo de nota", y="Árboles")

    st.subheader("🧠 Sistema experto (reglas IF/THEN)")
    expert_result = evaluate_movie(
        predicted_rating=chosen_prediction,
        award_wins=award_wins,
        award_nominations=award_nominations,
    )
    st.success(f"Conclusión: **{expert_result['recommendation']}**")
    st.write(expert_result["conclusion"])
    st.markdown("**Reglas activadas**")
    for rule in expert_result["fired_rules"]:
        st.markdown(f"- {rule}")
    st.caption(
        "La recomendación interpreta la nota del modelo y los datos de premios; "
        "no es una predicción adicional ni una opinión oficial."
    )

with st.expander("📊 Métricas completas y procedencia"):
    st.json(
        {
            "dataset": metadata["dataset_file"],
            "películas utilizadas": metadata["row_count"],
            "filas incompletas omitidas": metadata["skipped_missing"],
            "series omitidas": metadata["skipped_non_movies"],
            "MAE Árbol": metrics["decision_tree_mae"],
            "R² Árbol": metrics["decision_tree_r2"],
            "MAE Random Forest": metrics["random_forest_mae"],
            "R² Random Forest": metrics["random_forest_r2"],
            "R² OOB Random Forest": metrics["random_forest_oob_r2"],
            "MAE Red neuronal": metrics.get("neural_network_mae"),
            "R² Red neuronal": metrics.get("neural_network_r2"),
            "Umbral de pocos votos": metrics.get("low_votes_threshold"),
            "MAE red en películas con pocos votos": metrics.get(
                "neural_network_low_votes_mae"
            ),
            "Películas con pocos votos en prueba": metrics.get("low_votes_test_rows"),
        }
    )
