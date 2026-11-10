
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from gesture_features import (
    FEATURE_COLUMNS,
    GESTURES,
    NORMALIZATION,
)

DATA_PATH = Path("data/gestures_v2.csv")
MODEL_PATH = Path("models/gesture_model_v2.pkl")

RANDOM_STATE = 42


def make_model():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=22,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "No v2 dataset found. Run collect_data.py."
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "label",
        "session_id",
        *FEATURE_COLUMNS,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    if df[["label", "session_id"]].isna().any().any():
        raise ValueError(
            "Dataset contains missing labels or sessions"
        )

    unknown = set(df["label"]) - set(GESTURES)
    absent = set(GESTURES) - set(df["label"])

    if unknown:
        raise ValueError(
            f"Unexpected gesture labels: {unknown}"
        )

    if absent:
        raise ValueError(
            f"Missing gesture classes: {absent}"
        )

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    X = df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X).all():
        raise ValueError(
            "Dataset contains invalid feature values"
        )

    session_labels = df.groupby(
        "session_id"
    )["label"].nunique()

    if (session_labels > 1).any():
        raise ValueError(
            "One session ID contains multiple labels"
        )

    sessions_per_class = df.groupby(
        "label"
    )["session_id"].nunique()

    print("\nTotal samples:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nSessions per class:")
    print(sessions_per_class)

    if (sessions_per_class < 3).any():
        raise ValueError(
            "Record at least 3 separate sessions "
            "for every gesture before training."
        )

    return df


def split_by_session(df):
    rng = np.random.default_rng(RANDOM_STATE)

    test_sessions = set()

    for gesture in GESTURES:
        sessions = df.loc[
            df["label"] == gesture,
            "session_id"
        ].unique()

        sessions = rng.permutation(sessions)

        test_count = max(
            1,
            int(round(len(sessions) * 0.2))
        )

        test_sessions.update(
            sessions[:test_count].tolist()
        )

    is_test = df["session_id"].isin(test_sessions)

    train_df = df.loc[~is_test]
    test_df = df.loc[is_test]

    return train_df, test_df


def train_model():
    df = load_dataset()

    train_df, test_df = split_by_session(df)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )
    y_train = train_df["label"].to_numpy()

    X_test = test_df[FEATURE_COLUMNS].to_numpy(
        dtype=np.float32
    )
    y_test = test_df["label"].to_numpy()

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = make_model()

    print("\nTraining evaluation model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nSession-held-out accuracy: "
        f"{accuracy:.2%}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(GESTURES),
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print("Label order:", list(GESTURES))
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=list(GESTURES)
        )
    )

    print("\nTraining final model on all samples...")

    final_model = make_model()

    final_model.fit(
        df[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        ),
        df["label"].to_numpy()
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    bundle = {
        "model": final_model,
        "classes": list(final_model.classes_),
        "feature_columns": FEATURE_COLUMNS,
        "normalization": NORMALIZATION,
        "version": "gesture_controller_v2",
    }

    joblib.dump(bundle, MODEL_PATH)

    print("\nModel saved successfully!")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_model()
