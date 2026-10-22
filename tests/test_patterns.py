"""Controles du format et de l'ordre de lecture des pixels."""

from pathlib import Path
import tempfile
import unittest

from src.patterns import read_pattern


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


class ReadPatternTests(unittest.TestCase):
    def read_text_pattern(self, text):
        """Passe par un vrai fichier pour tester la lecture complete."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pattern.txt"
            path.write_text(text, encoding="utf-8")
            return read_pattern(path)

    def test_reference_patterns(self):
        for filename, target, active in (("zero.txt", 0, 16), ("un.txt", 1, 9)):
            with self.subTest(filename=filename):
                pixels, actual_target = read_pattern(DATA_DIRECTORY / filename)
                self.assertEqual(len(pixels), 48)
                self.assertEqual(actual_target, target)
                self.assertTrue(all(pixel in (0, 1) for pixel in pixels))
                self.assertEqual(sum(pixels), active)

    def test_reads_rows_from_left_to_right(self):
        rows = ["*.....", ".....*"] + ["......"] * 6
        pixels, target = self.read_text_pattern("\n".join(rows + ["1"]))
        expected = [1] + [0] * 10 + [1] + [0] * 36
        self.assertEqual(pixels, expected)
        self.assertEqual(target, 1)

    def test_accepts_windows_line_endings_and_final_newline(self):
        text = "\r\n".join(["......"] * 8 + ["0"]) + "\r\n"
        self.assertEqual(self.read_text_pattern(text), ([0] * 48, 0))

    def test_rejects_missing_and_extra_lines(self):
        for lines in (["......"] * 8, ["......"] * 8 + ["0", ""]):
            with self.subTest(lines=lines):
                with self.assertRaises(ValueError):
                    self.read_text_pattern("\n".join(lines) + "\n")

    def test_rejects_wrong_row_width(self):
        for row in (".....", "......."):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    self.read_text_pattern("\n".join([row] + ["......"] * 7 + ["0"]))

    def test_rejects_invalid_pixel(self):
        with self.assertRaises(ValueError):
            self.read_text_pattern("\n".join(["..x..."] + ["......"] * 7 + ["0"]))

    def test_rejects_invalid_target(self):
        for target in ("2", "zero", "0.0", " 1"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    self.read_text_pattern("\n".join(["......"] * 8 + [target]))


if __name__ == "__main__":
    unittest.main()
