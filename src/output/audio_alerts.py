"""Spoken audio alerts — Stage B optional module (CPU only, no torch).

Converts the nearest in-corridor obstacle into a spoken phrase via pyttsx3.
Rate-limited so the TTS engine is not called every frame (which would produce
garbled overlapping speech).

Example output: "obstacle, two metres, slightly left"

Install pyttsx3:  pip install pyttsx3
On Linux you may also need:  sudo apt-get install espeak
"""
from __future__ import annotations

import time
from typing import Optional

from src.obstacles.tracker import Track


def _metres_to_words(distance_m: float) -> str:
    """Convert a float distance in metres to a natural spoken phrase."""
    if distance_m < 1.0:
        return "less than one metre"
    if distance_m < 1.5:
        return "one metre"
    if distance_m < 2.5:
        return "two metres"
    if distance_m < 3.5:
        return "three metres"
    return f"{int(round(distance_m))} metres"


def _bearing_to_words(bearing_deg: float) -> str:
    """Convert a signed bearing in degrees to a directional phrase."""
    abs_b = abs(bearing_deg)
    if abs_b < 5:
        return "ahead"
    side = "right" if bearing_deg > 0 else "left"
    if abs_b < 20:
        return f"slightly {side}"
    if abs_b < 45:
        return side
    return f"far {side}"


class AudioAlerter:
    """Rate-limited TTS alerter.

    Args:
        min_interval_s: minimum seconds between two spoken alerts.
        rate: speech rate in words per minute (pyttsx3 default ~200).
    """

    def __init__(self, min_interval_s: float = 3.0, rate: int = 160) -> None:
        self._min_interval = min_interval_s
        self._last_alert_time = 0.0
        self._engine = None
        self._rate = rate

    def _ensure_engine(self) -> bool:
        """Lazily initialise pyttsx3 (returns False if unavailable)."""
        if self._engine is not None:
            return True
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            return True
        except Exception:
            return False

    def maybe_speak(self, tracks: list[Track]) -> Optional[str]:
        """Speak an alert for the nearest obstacle if the rate limit allows.

        Args:
            tracks: active tracks from the current frame (nearest-first order).

        Returns:
            The spoken phrase as a string, or None if no alert was issued.
        """
        if not tracks:
            return None

        now = time.monotonic()
        if now - self._last_alert_time < self._min_interval:
            return None

        nearest = tracks[0]  # caller should pass sorted list (nearest first)
        phrase = (
            f"obstacle, {_metres_to_words(nearest.distance_m)}, "
            f"{_bearing_to_words(nearest.bearing_deg)}"
        )

        if self._ensure_engine():
            self._engine.say(phrase)
            self._engine.runAndWait()

        self._last_alert_time = now
        return phrase


if __name__ == "__main__":
    from src.obstacles.tracker import Track
    import numpy as np

    alerter = AudioAlerter(min_interval_s=0.0)

    dummy_tracks = [
        Track(track_id=0, centroid_m=np.array([0.3, 0.0, 2.1]),
              distance_m=2.1, bearing_deg=8.0),
        Track(track_id=1, centroid_m=np.array([-0.5, 0.0, 4.0]),
              distance_m=4.0, bearing_deg=-7.0),
    ]

    phrase = alerter.maybe_speak(dummy_tracks)
    if phrase:
        print(f"Spoke: '{phrase}'")
    else:
        print("No alert issued (rate-limited or pyttsx3 unavailable).")
