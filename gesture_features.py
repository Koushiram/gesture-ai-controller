
import numpy as np


# ==========================================
# 1. GESTURE CLASSES
# ==========================================

GESTURES = (
    "THUMBS_UP",
    "PEACE",
    "OPEN_PALM",
    "FIST",
    "ROCK_SIGN",
    "NEUTRAL",
)


# ==========================================
# 2. FEATURE COLUMNS
# ==========================================

# MediaPipe detects 21 hand landmarks.
# Each landmark has x, y, z coordinates.
# Total features = 21 * 3 = 63.

FEATURE_COLUMNS = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]


# ==========================================
# 3. NORMALIZATION CONFIGURATION
# ==========================================

NORMALIZATION = "wrist_relative_palm_scale_v1"


# ==========================================
# 4. FEATURE EXTRACTION
# ==========================================

def extract_features(hand):
    """
    Convert MediaPipe's 21 hand landmarks
    into 63 normalized numerical features.

    Normalization:
    1. Use the wrist as the origin.
    2. Calculate palm size using landmarks 0 and 9.
    3. Normalize all landmark coordinates
       relative to wrist position and palm size.

    Returns:
        NumPy array containing 63 features.
    """

    if hand is None:
        raise ValueError(
            "No hand landmarks provided."
        )

    if len(hand) != 21:
        raise ValueError(
            f"Expected 21 landmarks, got {len(hand)}."
        )

    # Wrist landmark
    wrist = hand[0]

    # Middle finger MCP landmark
    palm = hand[9]

    # Calculate 3D palm size
    scale = (
        (palm.x - wrist.x) ** 2
        + (palm.y - wrist.y) ** 2
        + (palm.z - wrist.z) ** 2
    ) ** 0.5

    # Prevent division by zero
    scale = max(scale, 1e-6)

    features = []

    # Extract normalized coordinates
    for point in hand:
        normalized_x = (
            point.x - wrist.x
        ) / scale

        normalized_y = (
            point.y - wrist.y
        ) / scale

        normalized_z = (
            point.z - wrist.z
        ) / scale

        features.extend([
            normalized_x,
            normalized_y,
            normalized_z,
        ])

    # Convert to NumPy array
    features = np.asarray(
        features,
        dtype=np.float32
    )

    # Validate number of features
    if len(features) != 63:
        raise ValueError(
            "Feature extraction failed: "
            "expected 63 features."
        )

    # Check for NaN or infinity
    if not np.isfinite(features).all():
        raise ValueError(
            "Invalid landmark feature values."
        )

    return features


# ==========================================
# 5. TEST CONFIGURATION
# ==========================================

if __name__ == "__main__":

    print("=" * 45)
    print("AI GESTURE FEATURES V2")
    print("=" * 45)

    print("\nSupported gestures:")

    for gesture in GESTURES:
        print(" -", gesture)

    print("\nTotal gesture classes:", len(GESTURES))

    print(
        "Total feature columns:",
        len(FEATURE_COLUMNS)
    )

    print(
        "Normalization:",
        NORMALIZATION
    )

    print("\nFeature configuration ready!")
