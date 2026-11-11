"""Frame synchronisation with a linear chirp preamble.

A chirp has a sharp autocorrelation peak and spreads its energy across the
band, so a matched filter finds it reliably even when parts of the band are
attenuated by the speaker or notched by room echoes. The preamble is shared by
every modem, so sync quality does not depend on the modulation under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class Detection:
    #: sample index in the recording where the preamble starts
    start: int
    #: matched-filter peak divided by the correlation noise floor (a CFAR
    #: statistic: in pure noise it is roughly a standard normal deviate)
    score: float


class ChirpPreamble:
    def __init__(
        self,
        fs: int = 48_000,
        f0: float = 1_000.0,
        f1: float = 6_000.0,
        duration: float = 0.2,
        amplitude: float = 0.8,
        threshold: float = 7.0,
    ):
        """``duration`` sets the preamble energy and so the lowest SNR at which
        sync still works: the matched filter collects E = P * duration, so a
        200 ms chirp holds as much energy as 20 bits at 100 bit/s. ``threshold``
        is in units of the correlation noise floor; 7 keeps false alarms in
        pure noise below one in ~10^9 candidate positions.
        """
        self.fs = fs
        self.f0, self.f1 = f0, f1
        n = int(round(duration * fs))
        t = np.arange(n) / fs
        taper = signal.windows.tukey(n, 0.1)
        self.waveform = amplitude * taper * signal.chirp(t, f0, duration, f1, method="linear")
        self.threshold = threshold
        self._energy = float(np.sum(self.waveform**2))

    def __len__(self) -> int:
        return len(self.waveform)

    def correlate(self, rx: np.ndarray) -> np.ndarray:
        """Matched-filter output scaled by its own noise floor.

        The floor is estimated robustly from the median of |corr|, so the
        statistic does not depend on the recording level or on how long the
        recording is.
        """
        rx = np.asarray(rx, dtype=np.float64)
        if len(rx) < len(self.waveform):
            return np.zeros(0)
        corr = np.abs(signal.fftconvolve(rx, self.waveform[::-1], mode="valid"))
        floor = np.median(corr) / 0.6745
        return corr / max(floor, 1e-12 * np.sqrt(self._energy) + 1e-300)

    def detect(self, rx: np.ndarray) -> Detection | None:
        """Find the first (direct-path) arrival of the preamble.

        Echoes produce smaller correlation peaks *after* the direct path, so we
        take the earliest index whose score is close to the global maximum,
        rather than the maximum itself.
        """
        c = self.correlate(rx)
        if len(c) == 0:
            return None
        peak = float(np.max(c))
        if peak < self.threshold:
            return None
        # earliest sample within 80 % of the peak, then refine to its local max
        cand = int(np.argmax(c >= 0.8 * peak))
        hi = min(len(c), cand + int(0.002 * self.fs))
        start = cand + int(np.argmax(c[cand:hi]))
        return Detection(start=start, score=float(c[start]))
