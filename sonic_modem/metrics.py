"""BER measurement and theory curves."""

from __future__ import annotations

from math import comb

import numpy as np


def bit_errors(tx: np.ndarray, rx: np.ndarray | None) -> int:
    """Count bit errors; a missed frame counts every bit as wrong."""
    tx = np.asarray(tx)
    if rx is None:
        return len(tx)
    return int(np.count_nonzero(tx != np.asarray(rx)[: len(tx)]))


def snr_to_ebn0_db(snr_db, fs: float, bit_rate: float):
    """Convert full-band SNR (noise over 0..fs/2) to Eb/N0.

    With real white noise of variance s2 sampled at fs, N0 = 2*s2/fs and
    Eb = P/Rb, so Eb/N0 = SNR * fs / (2*Rb).
    """
    return np.asarray(snr_db) + 10 * np.log10(fs / (2 * bit_rate))


def ebn0_to_snr_db(ebn0_db, fs: float, bit_rate: float):
    return np.asarray(ebn0_db) - 10 * np.log10(fs / (2 * bit_rate))


def ber_noncoherent_mfsk(ebn0_db, M: int = 2) -> np.ndarray:
    """Theoretical BER of orthogonal M-FSK with non-coherent detection in AWGN."""
    k = int(np.log2(M))
    esn0 = k * 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    ps = np.zeros_like(esn0)
    for n in range(1, M):
        ps += (-1) ** (n + 1) * comb(M - 1, n) / (n + 1) * np.exp(-n / (n + 1) * esn0)
    return ps * (M / 2) / (M - 1)
