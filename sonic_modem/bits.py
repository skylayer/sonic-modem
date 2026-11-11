"""Bit helpers: bytes <-> bits, Gray coding."""

from __future__ import annotations

import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def random_bits(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.integers(0, 2, n, dtype=np.uint8)


def bits_to_ints(bits: np.ndarray, k: int) -> np.ndarray:
    """Group bits MSB-first into k-bit integers (len(bits) must be a multiple of k)."""
    b = np.asarray(bits, dtype=np.int64).reshape(-1, k)
    return b @ (1 << np.arange(k - 1, -1, -1))


def ints_to_bits(values: np.ndarray, k: int) -> np.ndarray:
    v = np.asarray(values, dtype=np.int64)[:, None]
    return ((v >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def gray_encode(n: np.ndarray) -> np.ndarray:
    n = np.asarray(n, dtype=np.int64)
    return n ^ (n >> 1)


def gray_decode(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.int64).copy()
    shift = g >> 1
    while np.any(shift):
        g ^= shift
        shift >>= 1
    return g
