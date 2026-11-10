
import math
import time
from collections import deque


class SnapDetector:
    def __init__(self):

        self.history = deque(maxlen=12)
        self.last_snap_time = 0

        self.cooldown = 1.5

    def distance(self, a, b):
        return math.hypot(
            a.x - b.x,
            a.y - b.y
        )

    def detect(self, hand):

        now = time.monotonic()

        thumb_tip = hand[4]
        middle_tip = hand[12]

        # Normalize distance using palm size
        palm_size = self.distance(
            hand[0], hand[9]
        )

        if palm_size < 0.001:
            return None

        finger_gap = (
            self.distance(
                thumb_tip, middle_tip
            ) / palm_size
        )

        self.history.append(
            (now, finger_gap)
        )

        # Need enough frames
        if len(self.history) < 4:
            return None

        # Find a recent pinch followed by separation
        recent = list(self.history)

        if now - self.last_snap_time < self.cooldown:
            return None

        for timestamp, gap in recent[:-1]:

            time_difference = now - timestamp

            # Fingers were touching recently
            was_pinched = gap < 0.30

            # Fingers are now separated
            is_released = finger_gap > 0.75

            # Movement happened quickly
            quick_motion = (
                0.03 < time_difference < 0.35
            )

            if (
                was_pinched
                and is_released
                and quick_motion
            ):

                self.last_snap_time = now
                self.history.clear()

                return "SNAP"

        return None

    def reset(self):
        self.history.clear()
