
from pathlib import Path

import joblib
import numpy as np

from gesture_features import (
    FEATURE_COLUMNS,
    GESTURES,
    NORMALIZATION,
    extract_features,
)


class MLGestureRecognizer:

    def __init__(
        self,
        model_path="models/gesture_model_v2.pkl",
        confidence_threshold=0.85,
        margin_threshold=0.12,
    ):
        path = Path(model_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}. "
                "Run python train_model.py first."
            )

        bundle = joblib.load(path)

        if bundle.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError(
                "Feature columns do not match"
            )

        if bundle.get("normalization") != NORMALIZATION:
            raise ValueError(
                "Model normalization does not match"
            )

        self.model = bundle["model"]
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

        if set(self.model.classes_) != set(GESTURES):
            raise ValueError(
                "Model classes do not match "
                "the six V2 gestures"
            )

    def predict(self, hand):
        if hand is None:
            return "UNKNOWN", 0.0

        features = extract_features(hand).reshape(
            1,
            -1
        )

        probabilities = self.model.predict_proba(
            features
        )[0]

        sorted_indices = np.argsort(
            probabilities
        )[::-1]

        best = int(sorted_indices[0])
        second = int(sorted_indices[1])

        gesture = str(self.model.classes_[best])
        confidence = float(probabilities[best])

        margin = (
            confidence
            - float(probabilities[second])
        )

        if (
            confidence < self.confidence_threshold
            or margin < self.margin_threshold
        ):
            return "UNKNOWN", confidence

        return gesture, confidence
