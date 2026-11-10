
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp

from action_controller import ActionController
from ml_gesture_recognizer import MLGestureRecognizer


LANDMARKER_PATH = Path("hand_landmarker.task")

HOLD_SECONDS = 0.65
RELEASE_SECONDS = 0.45
COOLDOWN_SECONDS = 1.2

CONFIDENCE_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.12

SMOOTHING_WINDOW = 5

ACTIVE_GESTURES = {
    "THUMBS_UP",
    "PEACE",
    "OPEN_PALM",
    "FIST",
    "ROCK_SIGN",
}


def create_landmarker():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(LANDMARKER_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    return (
        mp.tasks.vision.HandLandmarker
        .create_from_options(options)
    )


def draw_text(
    frame,
    text,
    y,
    color=(0, 255, 0)
):
    cv2.putText(
        frame,
        str(text),
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


def main():
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(
            "hand_landmarker.task is missing"
        )

    recognizer = MLGestureRecognizer(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        margin_threshold=MARGIN_THRESHOLD,
    )

    actions = ActionController()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam")

    automation_enabled = False
    armed = False

    candidate = None
    candidate_since = 0.0

    release_since = None
    last_action_time = -float("inf")

    history = deque(
        maxlen=SMOOTHING_WINDOW
    )

    last_action = "None"
    status = "Automation OFF"
    last_timestamp = -1

    print("\nAI HAND GESTURE CONTROLLER V2")
    print("E = Enable / Disable")
    print("Q = Quit")

    try:
        with create_landmarker() as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = max(
                    int(time.monotonic() * 1000),
                    last_timestamp + 1
                )
                last_timestamp = timestamp

                result = landmarker.detect_for_video(
                    image,
                    timestamp
                )

                hand = (
                    result.hand_landmarks[0]
                    if result.hand_landmarks
                    else None
                )

                if hand is None:
                    raw_gesture = "NO_HAND"
                    confidence = 0.0
                else:
                    raw_gesture, confidence = (
                        recognizer.predict(hand)
                    )

                history.append(raw_gesture)

                # Smooth predictions. Only accept
                # a gesture when the latest 5
                # predictions agree.
                if (
                    len(history) == SMOOTHING_WINDOW
                    and len(set(history)) == 1
                ):
                    gesture = history[-1]
                else:
                    gesture = "UNSTABLE"

                now = time.monotonic()

                if not automation_enabled:
                    armed = False
                    candidate = None
                    release_since = None
                    status = "Automation OFF"

                elif gesture in {
                    "NEUTRAL",
                    "NO_HAND"
                }:
                    candidate = None

                    if release_since is None:
                        release_since = now

                    if (
                        now - release_since
                        >= RELEASE_SECONDS
                    ):
                        armed = True
                        status = "Ready"

                else:
                    release_since = None

                    if not armed:
                        candidate = None
                        status = "Release to re-arm"

                    elif gesture not in ACTIVE_GESTURES:
                        candidate = None
                        status = "Waiting for stable gesture"

                    elif (
                        now - last_action_time
                        < COOLDOWN_SECONDS
                    ):
                        candidate = None
                        status = "Cooldown"

                    elif candidate != gesture:
                        candidate = gesture
                        candidate_since = now
                        status = "Hold gesture"

                    elif (
                        now - candidate_since
                        >= HOLD_SECONDS
                    ):
                        # Disarm before executing.
                        armed = False
                        candidate = None
                        last_action_time = now

                        action_success = (
                            actions.execute_action(gesture)
                        )

                        last_action = (
                            gesture
                            if action_success
                            else f"{gesture} FAILED"
                        )

                        status = "Release to re-arm"

                    else:
                        status = "Hold gesture"

                mode = (
                    "ON"
                    if automation_enabled
                    else "OFF"
                )

                draw_text(
                    frame,
                    f"Automation: {mode}",
                    35
                )

                draw_text(
                    frame,
                    f"Raw gesture: {raw_gesture}",
                    70
                )

                draw_text(
                    frame,
                    f"Stable gesture: {gesture}",
                    105
                )

                draw_text(
                    frame,
                    f"Confidence: {confidence:.1%}",
                    140
                )

                draw_text(
                    frame,
                    f"Status: {status}",
                    175
                )

                draw_text(
                    frame,
                    f"Last action: {last_action}",
                    210
                )

                draw_text(
                    frame,
                    "E = Toggle | Q = Quit",
                    245
                )

                cv2.imshow(
                    "AI Hand Gesture Controller V2",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("e"):
                    automation_enabled = (
                        not automation_enabled
                    )

                    armed = False
                    candidate = None
                    release_since = None
                    history.clear()

                    print(
                        "Automation:",
                        "ON"
                        if automation_enabled
                        else "OFF"
                    )

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
