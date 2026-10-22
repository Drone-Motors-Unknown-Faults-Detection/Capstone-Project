"""core.runner：共用 CLI 參數不接受前綴縮寫，save_json 拒絕 NaN（#269、#271）。"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.runner import add_dataset_args, add_openset_args, save_json


class DatasetArgsTests(unittest.TestCase):
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_dataset_args(parser)
        add_openset_args(parser)
        return parser

    def test_full_flag_is_accepted(self):
        args = self._parser().parse_args(["--data-root", "d2", "--seed", "7"])
        self.assertEqual((args.data_root, args.seed), ("d2", 7))

    def test_abbreviated_flag_is_rejected(self):
        parser = self._parser()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--data-r", "d2"])


class SaveJsonTests(unittest.TestCase):
    def test_numpy_scalars_and_arrays_are_serialised(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            save_json(path, {"i": np.int64(3), "f": np.float64(0.5), "a": np.arange(2), "none": None})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"i": 3, "f": 0.5, "a": [0, 1], "none": None})

    def test_nan_and_infinity_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            for value in (float("nan"), np.float64("nan"), float("inf"), np.float32("-inf")):
                with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                    save_json(path, {"metric": value})
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
