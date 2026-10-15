"""Application layer: the evaluation engine (``evaluate``) and its typed inputs."""

from cvpilot.eval.engine import EvalResult, eval_set_hash, evaluate
from cvpilot.eval.tables import AnnotationTable, ImageTable, PredictionTable

__all__ = [
    "AnnotationTable",
    "EvalResult",
    "ImageTable",
    "PredictionTable",
    "eval_set_hash",
    "evaluate",
]
