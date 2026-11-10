
import math


# ==========================================
# 1. DISTANCE CALCULATION
# ==========================================

def distance(a, b):
    """
    Calculate Euclidean distance between
    two MediaPipe hand landmarks.
    """
    return math.hypot(
        a.x - b.x,
        a.y - b.y
    )


# ==========================================
# 2. FINGER EXTENSION DETECTION
# ==========================================

def is_finger_extended(hand, tip, pip, mcp):
    """
    Determine whether a finger is extended.

    Uses joint distances to reduce
    dependency on hand orientation.
    """

    tip_to_mcp = distance(
        hand[tip], hand[mcp]
    )

    pip_to_mcp = distance(
        hand[pip], hand[mcp]
    )

    return (
        tip_to_mcp > pip_to_mcp * 1.5
    )


# ==========================================
# 3. THUMB EXTENSION DETECTION
# ==========================================

def is_thumb_extended(hand):
    """
    Check whether the thumb is extended.
    """

    thumb_tip = hand[4]
    thumb_ip = hand[3]
    thumb_mcp = hand[2]

    return (
        distance(thumb_tip, thumb_mcp)
        > distance(thumb_ip, thumb_mcp) * 1.5
    )


# ==========================================
# 4. NORMALIZED DISTANCE
# ==========================================

def normalized_distance(hand, a, b):
    """
    Normalize landmark distances by palm size.

    This helps recognition work at
    different distances from the camera.
    """

    palm_size = distance(
        hand[0], hand[9]
    )

    if palm_size < 0.001:
        return 0.0

    return (
        distance(hand[a], hand[b])
        / palm_size
    )


# ==========================================
# 5. MAIN GESTURE RECOGNITION
# ==========================================

def recognize_gesture(hand):
    """
    Recognize supported static gestures.

    Supported:
        THUMBS_UP
        PEACE
        OPEN_PALM
        FIST
        UNKNOWN

    Finger snapping is handled separately
    by dynamic_gesture_recognizer.py.
    """

    if hand is None or len(hand) != 21:
        return "UNKNOWN"

    # --------------------------------------
    # FINGER STATES
    # --------------------------------------

    index_extended = is_finger_extended(
        hand, 8, 6, 5
    )

    middle_extended = is_finger_extended(
        hand, 12, 10, 9
    )

    ring_extended = is_finger_extended(
        hand, 16, 14, 13
    )

    pinky_extended = is_finger_extended(
        hand, 20, 18, 17
    )

    thumb_extended = is_thumb_extended(hand)

    fingers = [
        index_extended,
        middle_extended,
        ring_extended,
        pinky_extended
    ]

    # --------------------------------------
    # THUMBS UP
    # --------------------------------------

    thumb_tip = hand[4]
    thumb_mcp = hand[2]
    wrist = hand[0]

    thumb_points_up = (
        thumb_tip.y < thumb_mcp.y
        and thumb_tip.y < wrist.y
    )

    if (
        thumb_extended
        and not any(fingers)
        and thumb_points_up
    ):
        return "THUMBS_UP"

    # --------------------------------------
    # PEACE SIGN
    # --------------------------------------

    if (
        index_extended
        and middle_extended
        and not ring_extended
        and not pinky_extended
    ):

        # Ensure index and middle fingers
        # are visibly separated
        finger_gap = normalized_distance(
            hand, 8, 12
        )

        if finger_gap > 0.30:
            return "PEACE"

    # --------------------------------------
    # OPEN PALM
    # --------------------------------------

    if (
        all(fingers)
        and thumb_extended
    ):

        # Thumb and index finger should
        # be sufficiently separated
        thumb_index_gap = normalized_distance(
            hand, 4, 8
        )

        # Thumb and middle finger should
        # not be in a snapping/pinch pose
        thumb_middle_gap = normalized_distance(
            hand, 4, 12
        )

        if (
            thumb_index_gap > 0.65
            and thumb_middle_gap > 0.65
        ):
            return "OPEN_PALM"

    # --------------------------------------
    # CLOSED FIST
    # --------------------------------------

    if (
        not any(fingers)
        and not thumb_extended
    ):

        # Ensure fingertips are close
        # to the palm
        index_folded = (
            normalized_distance(hand, 8, 0)
            < normalized_distance(hand, 6, 0)
        )

        middle_folded = (
            normalized_distance(hand, 12, 0)
            < normalized_distance(hand, 10, 0)
        )

        if index_folded and middle_folded:
            return "FIST"

    # --------------------------------------
    # NO MATCH
    # --------------------------------------

    return "UNKNOWN"
