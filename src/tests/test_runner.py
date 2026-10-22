"""core.runner：共用 CLI 參數不接受前綴縮寫（#269）。"""

from __future__ import annotations

import argparse
import contextlib
import io
import unittest

from core.runner import add_dataset_args, add_openset_args


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


if __name__ == "__main__":
    unittest.main()
