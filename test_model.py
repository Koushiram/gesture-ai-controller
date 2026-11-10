
from pathlib import Path

import joblib

from gesture_features import (
    FEATURE_COLUMNS,
    GESTURES,
    NORMALIZATION,
)


def main():
    model_path = Path("models/gesture_model_v2.pkl")

    print("=" * 45)
    print("AI HAND GESTURE MODEL V2 - TEST")
    print("=" * 45)

    if not model_path.exists():
        print("ERROR: Trained model not found!")
        print("Run: python train_model.py")
        return

    bundle = joblib.load(model_path)
    model = bundle["model"]

    print("\nModel loaded successfully!")
    print("Model version:", bundle.get("version"))
    print("Number of features:", len(FEATURE_COLUMNS))

    print("\nTrained gesture classes:")
    for gesture in model.classes_:
        print(" -", gesture)

    checks = {
        "Feature columns": (
            bundle.get("feature_columns")
            == FEATURE_COLUMNS
        ),
        "Normalization": (
            bundle.get("normalization")
            == NORMALIZATION
        ),
        "Gesture classes": (
            set(model.classes_) == set(GESTURES)
        ),
    }

    print("\nValidation results:")

    for name, passed in checks.items():
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    if all(checks.values()):
        print("\nSUCCESS: Model is ready!")
        print("Run: python hand_detector.py")
    else:
        print("\nModel validation failed.")
        print("Check your training configuration.")


if __name__ == "__main__":
    main()
