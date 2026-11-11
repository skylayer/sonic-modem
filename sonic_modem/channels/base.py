"""Channel interface.

Anything that takes a transmit waveform and returns what the receiver hears is
a channel: the simulated room below, or later a real speaker and microphone
(:mod:`sonic_modem.channels.audio`). The receiver must not assume the output has
the same length or alignment as the input.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Channel(ABC):
    fs: int

    @abstractmethod
    def transmit(self, samples: np.ndarray) -> np.ndarray:
        """Play ``samples`` and return the recording."""


class IdealChannel(Channel):
    """Wire: output equals input. Useful for loopback tests."""

    def __init__(self, fs: int = 48_000):
        self.fs = fs

    def transmit(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(samples, dtype=np.float64).copy()
