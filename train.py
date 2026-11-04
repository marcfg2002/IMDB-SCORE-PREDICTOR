"""Entrena modelos de regresión para estimar la nota IMDb."""

from __future__ import annotations

import argparse
import csv
import io
import math
from collections import Counter
from pathlib import Path
from typing import Any, Protocol
from zipfile import BadZipFile, ZipFile

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
LOW_VOTES_QUANTILE = 0.25
DEFAULT_DATASET_ZIP = Path("data") / "archive (1).zip"
DEFAULT_CSV_NAME = "mflix.movies.csv"
DEFAULT_MODEL_PATH = Path("modelo.joblib")
NUMERIC_COLUMNS = {
    "year": "year",
    "runtime": "runtime",
    "imdb_votes": "imdb.votes",
    "awards_wins": "awards.wins",
    "awards_nominations": "awards.nominations",
}
GENRE_COLUMN = "genres[0]"
TARGET_COLUMN = "imdb.rating"
REQUIRED_COLUMNS = {*NUMERIC_COLUMNS.values(), GENRE_COLUMN, TARGET_COLUMN}


class RatingPredictor(Protocol):
    def predict(self, feature_rows: list[dict[str, Any]]) -> np.ndarray: ...


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena modelos de regresión para estimar la nota IMDb de películas Mflix."
    )
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=DEFAULT_DATASET_ZIP,
        help=f"ZIP del dataset Mflix. Por defecto: {DEFAULT_DATASET_ZIP}",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=DEFAULT_CSV_NAME,
        help=f"CSV dentro del ZIP que se usará para entrenar. Por defecto: {DEFAULT_CSV_NAME}",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ruta del artefacto de salida. Por defecto: ./modelo.joblib",
    )
    return parser.parse_args()


def _number_or_none(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "")
    if not value or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def load_dataset(
    dataset_zip: Path, csv_name: str
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Lee el CSV directamente desde el ZIP y omite filas sin datos utilizables."""
    if not dataset_zip.is_file():
        raise FileNotFoundError(f"No existe el archivo ZIP indicado: {dataset_zip}")

    try:
        archive = ZipFile(dataset_zip)
    except BadZipFile as error:
        raise ValueError(f"El archivo no es un ZIP válido: {dataset_zip}") from error

    with archive:
        if csv_name not in archive.namelist():
            available_csvs = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            raise FileNotFoundError(
                f"No se encontró {csv_name!r} dentro del ZIP. CSV disponibles: {available_csvs}"
            )

        feature_rows: list[dict[str, Any]] = []
        targets: list[float] = []
        skipped_missing = 0
        skipped_non_movies = 0
        with archive.open(csv_name) as binary_file:
            text_file = io.TextIOWrapper(binary_file, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_file)
            available_columns = set(reader.fieldnames or [])
            missing_columns = sorted(REQUIRED_COLUMNS - available_columns)
            if missing_columns:
                raise ValueError(
                    f"El CSV {csv_name!r} no contiene las columnas necesarias: {missing_columns}"
                )

            for row in reader:
                if row.get("type", "").strip().lower() not in ("", "movie"):
                    skipped_non_movies += 1
                    continue

                rating = _number_or_none(row, TARGET_COLUMN)
                genre = row.get(GENRE_COLUMN, "").strip()
                values = {
                    feature_name: _number_or_none(row, column)
                    for feature_name, column in NUMERIC_COLUMNS.items()
                }
                if rating is None or not genre or any(value is None for value in values.values()):
                    skipped_missing += 1
                    continue
                if not 0 <= rating <= 10:
                    skipped_missing += 1
                    continue

                feature_rows.append({**values, "genre": genre})
                targets.append(rating)

    if len(feature_rows) < 10:
        raise ValueError(
            f"Solo se encontraron {len(feature_rows)} filas válidas; se necesitan al menos 10."
        )

    numeric_values = {
        name: np.asarray([float(row[name]) for row in feature_rows], dtype=np.float64)
        for name in NUMERIC_COLUMNS
    }
    genre_counts = Counter(str(row["genre"]) for row in feature_rows)
    metadata = {
        "dataset_type": "mflix-movie-rating-regression",
        "dataset_file": csv_name,
        "target": TARGET_COLUMN,
        "feature_names": list(NUMERIC_COLUMNS) + ["genre"],
        "row_count": len(feature_rows),
        "skipped_missing": skipped_missing,
        "skipped_non_movies": skipped_non_movies,
        "genres": sorted(genre_counts),
        "default_genre": genre_counts.most_common(1)[0][0],
        "feature_ranges": {
            name: {"min": float(values.min()), "max": float(values.max())}
            for name, values in numeric_values.items()
        },
        "feature_defaults": {
            name: float(np.median(values)) for name, values in numeric_values.items()
        },
    }
    return feature_rows, np.asarray(targets, dtype=np.float64), metadata


def make_models() -> tuple[Pipeline, Pipeline, Pipeline]:
    """Crea modelos de referencia y una red neuronal para regresión."""
    decision_tree = Pipeline(
        [
            ("vectorizer", DictVectorizer()),
            (
                "regressor",
                DecisionTreeRegressor(
                    criterion="squared_error",
                    max_depth=12,
                    min_samples_leaf=10,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        [
            ("vectorizer", DictVectorizer()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=160,
                    criterion="squared_error",
                    max_depth=16,
                    max_features=0.8,
                    min_samples_leaf=5,
                    bootstrap=True,
                    oob_score=True,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    neural_network = Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
            (
                "regressor",
                MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    n_iter_no_change=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return decision_tree, random_forest, neural_network


def evaluate_low_votes(
    model: RatingPredictor,
    feature_rows: list[dict[str, Any]],
    targets: np.ndarray,
    threshold: float,
) -> tuple[float | None, int]:
    """Evalúa la red en películas del cuartil inferior de votos."""
    low_votes_mask = np.asarray(
        [float(row["imdb_votes"]) <= threshold for row in feature_rows],
        dtype=bool,
    )
    sample_count = int(low_votes_mask.sum())
    if sample_count == 0:
        return None, 0

    low_vote_rows = [row for row, is_low in zip(feature_rows, low_votes_mask) if is_low]
    predictions = np.clip(model.predict(low_vote_rows), 0, 10)
    return float(mean_absolute_error(targets[low_votes_mask], predictions)), sample_count


def main() -> None:
    args = parse_args()
    print(f"[1/6] Leyendo {args.csv_name} desde {args.dataset_zip}...")
    feature_rows, targets, metadata = load_dataset(args.dataset_zip, args.csv_name)
    print(
        f"Películas utilizables: {metadata['row_count']}; "
        f"filas omitidas por datos incompletos: {metadata['skipped_missing']}; "
        f"series omitidas: {metadata['skipped_non_movies']}."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        feature_rows,
        targets,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    decision_tree, random_forest, neural_network = make_models()

    print("[2/6] Entrenando Árbol de Decisión...")
    decision_tree.fit(X_train, y_train)
    print("[3/6] Entrenando Random Forest con bootstrap y OOB...")
    random_forest.fit(X_train, y_train)
    print("[4/6] Entrenando Red neuronal MLP...")
    neural_network.named_steps["regressor"].set_params(
        early_stopping=len(X_train) >= 30
    )
    neural_network.fit(X_train, y_train)

    dt_predictions = decision_tree.predict(X_test)
    rf_predictions = random_forest.predict(X_test)
    nn_predictions = np.clip(neural_network.predict(X_test), 0, 10)
    low_votes_threshold = float(
        np.quantile([float(row["imdb_votes"]) for row in X_train], LOW_VOTES_QUANTILE)
    )
    low_votes_mae, low_votes_test_rows = evaluate_low_votes(
        neural_network, X_test, y_test, low_votes_threshold
    )
    metrics = {
        "decision_tree_mae": float(mean_absolute_error(y_test, dt_predictions)),
        "decision_tree_r2": float(r2_score(y_test, dt_predictions)),
        "random_forest_mae": float(mean_absolute_error(y_test, rf_predictions)),
        "random_forest_r2": float(r2_score(y_test, rf_predictions)),
        "random_forest_oob_r2": float(
            random_forest.named_steps["regressor"].oob_score_
        ),
        "neural_network_mae": float(mean_absolute_error(y_test, nn_predictions)),
        "neural_network_r2": float(r2_score(y_test, nn_predictions)),
        "neural_network_low_votes_mae": low_votes_mae,
        "low_votes_threshold": low_votes_threshold,
        "low_votes_test_rows": low_votes_test_rows,
        "test_rows": int(len(y_test)),
    }
    metadata["metrics"] = metrics

    print("[5/6] Rendimiento en el conjunto de prueba:")
    print(f"  Árbol - MAE: {metrics['decision_tree_mae']:.3f} | R²: {metrics['decision_tree_r2']:.3f}")
    print(f"  Bosque - MAE: {metrics['random_forest_mae']:.3f} | R²: {metrics['random_forest_r2']:.3f}")
    print(f"  Red neuronal - MAE: {metrics['neural_network_mae']:.3f} | R²: {metrics['neural_network_r2']:.3f}")
    if low_votes_mae is None:
        print("  Prueba de pocos votos: no hubo películas en ese subconjunto de prueba.")
    else:
        print(
            f"  Prueba de pocos votos (<= {low_votes_threshold:.0f}): "
            f"MAE {low_votes_mae:.3f} en {low_votes_test_rows} películas."
        )
    print(f"  OOB Random Forest (R²): {metrics['random_forest_oob_r2']:.3f}")

    artifact = {
        "artifact_version": 3,
        "decision_tree": decision_tree,
        "random_forest": random_forest,
        "neural_network": neural_network,
        "metadata": metadata,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path, compress=3)
    print(f"[6/6] Modelo guardado en: {args.model_path.resolve()}")


if __name__ == "__main__":
    main()
