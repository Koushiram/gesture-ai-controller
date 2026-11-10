
import csv
import time
import uuid
from pathlib import Path

import cv2
import mediapipe as mp

from gesture_features import (
    FEATURE_COLUMNS,
    extract_features,
)

DATA_PATH = Path("data/gestures_v2.csv")
LANDMARKER_PATH = Path("hand_landmarker.task")

SAMPLES_PER_SESSION = 100
CAPTURE_INTERVAL = 0.10

GESTURE_KEYS = {
    ord("1"): "THUMBS_UP",
    ord("2"): "PEACE",
    ord("3"): "OPEN_PALM",
    ord("4"): "FIST",
    ord("5"): "ROCK_SIGN",
    ord("6"): "NEUTRAL",
}

HEADER = [
    "label",
    "session_id",
    *FEATURE_COLUMNS,
]


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


def initialize_dataset():
    DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if DATA_PATH.exists():
        with DATA_PATH.open("r", newline="") as file:
            existing_header = next(
                csv.reader(file),
                None
            )

        if existing_header != HEADER:
            raise ValueError(
                "Existing dataset has an incompatible "
                "header. Back it up before replacing it."
            )
    else:
        with DATA_PATH.open("w", newline="") as file:
            csv.writer(file).writerow(HEADER)


def main():
    if not LANDMARKER_PATH.exists():
        raise FileNotFoundError(
            "hand_landmarker.task is missing"
        )

    initialize_dataset()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Cannot open webcam")

    current_label = None
    session_id = None
    sample_count = 0
    start_time = 0.0
    last_capture = 0.0
    last_timestamp = -1

    print("\nGESTURE DATA COLLECTION V2")
    print("1 - THUMBS_UP")
    print("2 - PEACE")
    print("3 - OPEN_PALM")
    print("4 - FIST")
    print("5 - ROCK_SIGN")
    print("6 - NEUTRAL")
    print("Q - Quit")
    print("\nEach recording saves 100 samples.")

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

                now = time.monotonic()

                if current_label is not None:
                    if (
                        now >= start_time
                        and hand is not None
                        and now - last_capture
                        >= CAPTURE_INTERVAL
                    ):
                        features = extract_features(hand)

                        with DATA_PATH.open(
                            "a",
                            newline=""
                        ) as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                current_label,
                                session_id,
                                *features.tolist()
                            ])

                        sample_count += 1
                        last_capture = now

                        if sample_count >= SAMPLES_PER_SESSION:
                            print(
                                f"Saved {sample_count} samples "
                                f"for {current_label}"
                            )
                            current_label = None

                if current_label is None:
                    status = "Press 1-6 to record"
                elif now < start_time:
                    status = (
                        f"Prepare {current_label}: "
                        f"{start_time-now:.1f}s"
                    )
                else:
                    status = (
                        f"{current_label}: "
                        f"{sample_count}/{SAMPLES_PER_SESSION}"
                    )

                cv2.putText(
                    frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Q = Quit",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Gesture Collection V2",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key in GESTURE_KEYS:
                    current_label = GESTURE_KEYS[key]
                    session_id = uuid.uuid4().hex
                    sample_count = 0
                    last_capture = 0.0
                    start_time = time.monotonic() + 2.0

                    print(
                        f"Starting {current_label} "
                        "in 2 seconds..."
                    )

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
