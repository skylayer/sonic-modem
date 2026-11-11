"""Common interface for all modulation schemes (ASK, FSK, PSK, QAM, ...).

A modem only maps bits to a baseband-free *audio* waveform and back. It knows
nothing about synchronisation or the channel: :class:`sonic_modem.link.Link`
finds the start of the data with a preamble and hands the modem a buffer whose
first sample is the first sample of the first symbol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Modem(ABC):
    #: sample rate in Hz
    fs: int
    #: samples per symbol
    samples_per_symbol: int
    #: bits carried by one symbol
    bits_per_symbol: int

    @property
    def symbol_rate(self) -> float:
        return self.fs / self.samples_per_symbol

    @property
    def bit_rate(self) -> float:
        return self.symbol_rate * self.bits_per_symbol

    def n_symbols(self, n_bits: int) -> int:
        return -(-n_bits // self.bits_per_symbol)

    def n_samples(self, n_bits: int) -> int:
        return self.n_symbols(n_bits) * self.samples_per_symbol

    def pad_bits(self, bits: np.ndarray) -> np.ndarray:
        """Zero-pad ``bits`` to a whole number of symbols."""
        bits = np.asarray(bits, dtype=np.uint8)
        extra = (-len(bits)) % self.bits_per_symbol
        return np.concatenate([bits, np.zeros(extra, np.uint8)]) if extra else bits

    @abstractmethod
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Return a float64 waveform in [-1, 1] carrying ``bits``."""

    @abstractmethod
    def demodulate(self, samples: np.ndarray, n_bits: int) -> np.ndarray:
        """Recover ``n_bits`` bits from ``samples`` aligned to the first symbol."""

    def describe(self) -> str:
        return f"{type(self).__name__}: {self.bit_rate:.1f} bit/s, {self.symbol_rate:.1f} baud"
