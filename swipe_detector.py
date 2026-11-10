
import math
import time
from collections import deque


class SwipeDetector:
    """
    Detects horizontal movement of two extended fingers.

    The caller must pass a hand only when the
    ML model recognizes PEACE.
    """

    def __init__(
        self,
        window_seconds=0.65,
        min_distance=1.35,
        max_vertical_ratio=0.45,
        cooldown=1.5
    ):
        self.window_seconds = window_seconds
        self.min_distance = min_distance
        self.max_vertical_ratio = max_vertical_ratio
        self.cooldown = cooldown

        self.history = deque(maxlen=40)
        self.last_swipe_time = float("-inf")

    @staticmethod
    def distance(a, b):
        return math.hypot(
            a.x - b.x,
            a.y - b.y
        )

    def reset(self):
        self.history.clear()

    def detect(self, hand):
        now = time.monotonic()

        if now - self.last_swipe_time < self.cooldown:
            self.reset()
            return None

        if hand is None or len(hand) != 21:
            self.reset()
            return None

        # Palm size provides scale invariance.
        palm_size = self.distance(hand[0], hand[9])

        if palm_size < 0.001:
            self.reset()
            return None

        # Midpoint of index and middle fingertips.
        x = (hand[8].x + hand[12].x) / 2
        y = (hand[8].y + hand[12].y) / 2

        self.history.append((now, x, y, palm_size))

        # Keep only recent positions.
        while (
            self.history
            and now - self.history[0][0]
            > self.window_seconds
        ):
            self.history.popleft()

        if len(self.history) < 4:
            return None

        start_time, start_x, start_y, start_scale = (
            self.history[0]
        )

        elapsed = now - start_time

        if elapsed < 0.12:
            return None

        average_scale = max(
            (start_scale + palm_size) / 2,
            0.001
        )

        dx = (x - start_x) / average_scale
        dy = (y - start_y) / average_scale

        # Require substantial horizontal movement.
        if abs(dx) < self.min_distance:
            return None

        # Reject predominantly vertical movement.
        if abs(dy) > (
            abs(dx) * self.max_vertical_ratio
        ):
            return None

        # Confirm the movement generally travels
        # in the same horizontal direction.
        positions = list(self.history)
        direction = 1 if dx > 0 else -1

        movement = [
            (positions[i][1] - positions[i - 1][1])
            * direction
            for i in range(1, len(positions))
        ]

        forward = sum(
            max(step, 0) for step in movement
        )

        backward = sum(
            max(-step, 0) for step in movement
        )

        if forward <= 0:
            return None

        if backward > forward * 0.35:
            return None

        self.last_swipe_time = now
        self.reset()

        # Camera is mirrored in hand_detector.py.
        # Positive X means movement to screen right.
        if dx > 0:
            return "SWIPE_RIGHT"

        return "SWIPE_LEFT"
