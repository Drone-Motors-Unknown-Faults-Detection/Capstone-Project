"""最小歷史矩陣；來源固定為 3aa554f，並非新的實測結果。"""

from copy import deepcopy

SOURCE = ("https://github.com/Drone-Motors-Unknown-Faults-Detection/Capstone-Project/blob/"
          "3aa554f73932de3545ae6a69ac37113575477e11/"
          "output/exp12_confusion_tsne/2026-10-09-14-27-55/summary.json")


def historical_result() -> dict:
    labels = ["8screws", "1screw", "2screws", "3screws", "4screws", "unknown"]
    return deepcopy({
        "seed": 42, "confidence": 0.95, "perplexity": 30.0, "n_datasets": 9,
        "binary_confusion": {"labels": ["healthy", "fault"], "matrix": [[527, 45], [0, 26070]]},
        "openset_confusion": {
            "ledoit_wolf": {"labels": labels, "matrix": [
                [525, 0, 0, 0, 0, 47], [0, 562, 0, 0, 0, 39], [0, 0, 559, 0, 0, 46],
                [0, 0, 0, 561, 0, 30], [0, 0, 0, 0, 529, 40], [0, 0, 0, 0, 0, 14381]]},
            "legacy": {"labels": labels, "matrix": [
                [0, 0, 0, 0, 0, 572], [0, 0, 0, 0, 0, 601], [0, 0, 0, 0, 0, 605],
                [0, 0, 0, 0, 0, 591], [0, 0, 0, 0, 0, 569], [0, 0, 0, 0, 0, 14381]]},
        },
    })
