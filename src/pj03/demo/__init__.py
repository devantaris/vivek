"""Demonstrable product: a self-contained hate-speech classifier demo.

Wraps the trained TF-IDF+LogReg baseline into an API that returns a label,
a probability, a model-grounded rationale (the n-grams that pushed the
decision, with their weights), and live faithfulness scores.
"""

from pj03.demo.predictor import DemoPredictor, PredictionResult

__all__ = ["DemoPredictor", "PredictionResult"]
