"""M-ary frequency shift keying with non-coherent (energy) detection."""

from __future__ import annotations

import numpy as np

from ..bits import bits_to_ints, gray_decode, gray_encode, ints_to_bits
from .base import Modem


class FSKModem(Modem):
    """Continuous-phase M-FSK.

    Each symbol is a tone chosen from ``M`` frequencies centred on
    ``center_freq`` and ``tone_spacing`` apart. Bits map to tones through a
    Gray code, so a mistake between neighbouring tones costs one bit.

    The receiver is non-coherent: for every symbol it correlates against each
    tone and picks the one with the most energy, so it needs no carrier phase.
    ``guard`` skips the first part of every symbol before integrating, which
    lets room echoes from the previous symbol die out (a cheap defence against
    inter-symbol interference) at the cost of ``-10 log10(1 - guard)`` dB of
    collected energy.

    For the tones to be orthogonal inside the integration window,
    ``tone_spacing`` should be a multiple of ``1 / window``; a warning-free
    default is chosen when ``tone_spacing`` is ``None``.
    """

    def __init__(
        self,
        fs: int = 48_000,
        baud: float = 100.0,
        M: int = 2,
        center_freq: float = 2_500.0,
        tone_spacing: float | None = None,
        guard: float = 0.2,
        amplitude: float = 0.8,
    ):
        if M < 2 or M & (M - 1):
            raise ValueError("M must be a power of two >= 2")
        if not 0 <= guard < 1:
            raise ValueError("guard must be in [0, 1)")
        sps = fs / baud
        if abs(sps - round(sps)) > 1e-9:
            raise ValueError("fs / baud must be an integer number of samples")
        self.fs = int(fs)
        self.samples_per_symbol = int(round(sps))
        self.M = M
        self.bits_per_symbol = M.bit_length() - 1
        self.guard_samples = int(round(guard * self.samples_per_symbol))
        self.window = self.samples_per_symbol - self.guard_samples
        if tone_spacing is None:
            # smallest multiple of 1/window that is also >= 2*baud (gives some
            # margin against Doppler / clock drift)
            base = self.fs / self.window
            tone_spacing = base * max(1, int(np.ceil(2 * baud / base)))
        self.tone_spacing = float(tone_spacing)
        self.center_freq = float(center_freq)
        self.amplitude = amplitude
        self.tones = center_freq + tone_spacing * (np.arange(M) - (M - 1) / 2)
        if self.tones[0] <= 0 or self.tones[-1] >= fs / 2:
            raise ValueError(f"tones {self.tones} do not fit in (0, fs/2)")

        # correlation bank over the integration window, shape (M, window)
        n = np.arange(self.guard_samples, self.samples_per_symbol)
        self._bank = np.exp(-2j * np.pi * np.outer(self.tones, n) / self.fs)

    # ------------------------------------------------------------------ tx
    def symbols_from_bits(self, bits: np.ndarray) -> np.ndarray:
        return gray_encode(bits_to_ints(self.pad_bits(bits), self.bits_per_symbol))

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        sym = self.symbols_from_bits(bits)
        freq = np.repeat(self.tones[sym], self.samples_per_symbol)
        # continuous phase: integrate instantaneous frequency, starting at 0
        phase = 2 * np.pi * (np.cumsum(freq) - freq) / self.fs
        return self.amplitude * np.sin(phase)

    # ------------------------------------------------------------------ rx
    def tone_energies(self, samples: np.ndarray, n_symbols: int) -> np.ndarray:
        """|correlation|^2 per (symbol, tone), shape (n_symbols, M)."""
        need = n_symbols * self.samples_per_symbol
        x = np.asarray(samples, dtype=np.float64)[:need]
        if len(x) < need:
            x = np.concatenate([x, np.zeros(need - len(x))])
        frames = x.reshape(n_symbols, self.samples_per_symbol)[:, self.guard_samples :]
        return np.abs(frames @ self._bank.T) ** 2

    def demodulate(self, samples: np.ndarray, n_bits: int) -> np.ndarray:
        n_sym = self.n_symbols(n_bits)
        sym = np.argmax(self.tone_energies(samples, n_sym), axis=1)
        return ints_to_bits(gray_decode(sym), self.bits_per_symbol)[:n_bits]

    def describe(self) -> str:
        tones = ", ".join(f"{f:.0f}" for f in self.tones)
        return f"{self.M}-FSK {super().describe().split(': ', 1)[1]}, tones [{tones}] Hz"
