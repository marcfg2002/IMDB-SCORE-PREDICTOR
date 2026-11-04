import csv
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from train import evaluate_low_votes, load_dataset


class ConstantRatingModel:
    def predict(self, feature_rows: list[dict[str, object]]) -> np.ndarray:
        return np.full(len(feature_rows), 6.0)


class DatasetLoadingTests(unittest.TestCase):
    def test_load_dataset_keeps_complete_movies_and_skips_invalid_rows(self) -> None:
        columns = [
            "type",
            "year",
            "runtime",
            "imdb.votes",
            "awards.wins",
            "awards.nominations",
            "genres[0]",
            "imdb.rating",
        ]
        rows = [
            {
                "type": "movie",
                "year": str(2000 + index),
                "runtime": "100",
                "imdb.votes": "250",
                "awards.wins": "1",
                "awards.nominations": "2",
                "genres[0]": "Drama",
                "imdb.rating": str(6 + index / 10),
            }
            for index in range(10)
        ]
        rows.append({**rows[0], "imdb.rating": ""})
        rows.append({**rows[0], "type": "series"})

        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "mflix.zip"
            with ZipFile(archive_path, "w") as archive:
                with tempfile.TemporaryFile(mode="w+", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
                    csv_file.seek(0)
                    archive.writestr("mflix.movies.csv", csv_file.read())

            features, targets, metadata = load_dataset(archive_path, "mflix.movies.csv")

        self.assertEqual(len(features), 10)
        self.assertEqual(len(targets), 10)
        self.assertTrue(all(feature["genre"] == "Drama" for feature in features))
        self.assertEqual(metadata["skipped_missing"], 1)
        self.assertEqual(metadata["skipped_non_movies"], 1)

    def test_low_votes_evaluation_measures_only_movies_below_threshold(self) -> None:
        feature_rows = [
            {"imdb_votes": 5.0},
            {"imdb_votes": 10.0},
            {"imdb_votes": 20.0},
        ]
        targets = np.asarray([7.0, 5.0, 8.0])

        mae, sample_count = evaluate_low_votes(
            ConstantRatingModel(),
            feature_rows,
            targets,
            threshold=10.0,
        )

        self.assertEqual(mae, 1.0)
        self.assertEqual(sample_count, 2)

    def test_low_votes_evaluation_handles_an_empty_test_subset(self) -> None:
        mae, sample_count = evaluate_low_votes(
            ConstantRatingModel(),
            [{"imdb_votes": 20.0}],
            np.asarray([8.0]),
            threshold=10.0,
        )

        self.assertIsNone(mae)
        self.assertEqual(sample_count, 0)


if __name__ == "__main__":
    unittest.main()
