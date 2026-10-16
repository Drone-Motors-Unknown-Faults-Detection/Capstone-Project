"""Unknown/uncertain diagnosis policy layered on the Open Set result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DiagnosisDecision:
    fault_type: str
    confidence: float | None
    is_unknown: bool


class DiagnosisResolver:
    """Resolve a detector label without inventing a physical cause.

    ``label_to_fault_type`` is intentionally optional.  With no versioned
    taxonomy, known predictions remain ``uncertain``; unknown Open Set scores
    always remain ``unknown`` even if a label mapping was supplied.
    """

    def __init__(self, label_to_fault_type: Mapping[int, str] | None = None) -> None:
        self.label_to_fault_type = {
            int(label): str(name)
            for label, name in (label_to_fault_type or {}).items()
            if str(name).strip()
        }

    def resolve(self, predicted_label: int, openset_score: float, decision_confidence: float) -> DiagnosisDecision:
        if openset_score < 0.0:
            raise ValueError("openset_score must be non-negative")
        if not 0.0 <= decision_confidence <= 1.0:
            raise ValueError("decision_confidence must be in [0, 1]")
        if int(predicted_label) < 0 or float(openset_score) > 1.0:
            return DiagnosisDecision("unknown", None, True)
        name = self.label_to_fault_type.get(int(predicted_label))
        if name is None:
            return DiagnosisDecision("uncertain", None, False)
        return DiagnosisDecision(name, float(decision_confidence), False)
