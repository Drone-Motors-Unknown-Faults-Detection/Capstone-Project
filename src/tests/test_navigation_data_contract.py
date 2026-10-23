"""experiments.navigation_data_contract：來源 SHA 讀實際目錄，別名目錄也要有雜湊（#301 F-03）。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.feature_schema import SCHEMAS
from core.provenance import sha256_file
from experiments.navigation_data_contract import run


class AliasDirectoryTests(unittest.TestCase):
    def test_alias_directory_sources_are_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "Step-1/myfeature/T1/8000rpm"
            rng = np.random.default_rng(1)
            for folder, center in (("8screws", 0.0), ("1screw", 3.0)):
                (base / folder).mkdir(parents=True)
                pd.DataFrame(rng.normal(center, 0.1, (20, 105)), columns=SCHEMAS["historical_clean_v1"]).to_csv(
                    base / folder / "T1_Group_feature_data_clean.csv", index=False)
            result = run({"motor": "T1", "rpm": "8000rpm", "path": base})
            expected = sha256_file(base / "1screw" / "T1_Group_feature_data_clean.csv")
        self.assertEqual(result["sources"]["1screws"],
                         [{"sha256": expected, "path": "Step-1/myfeature/T1/8000rpm/1screw/T1_Group_feature_data_clean.csv"}])
        config = next(item for item in result["configs"] if item["config"] == "1screws")
        self.assertEqual(config["sha256"], [expected])
        self.assertTrue(all(sources for sources in result["sources"].values()))


if __name__ == "__main__":
    unittest.main()
