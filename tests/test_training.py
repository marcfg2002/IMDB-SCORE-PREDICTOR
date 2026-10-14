import csv
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from train import load_dataset


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


if __name__ == "__main__":
    unittest.main()
