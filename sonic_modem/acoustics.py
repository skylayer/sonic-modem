"""Physical acoustics helpers: atmospheric absorption of sound (ISO 9613-1)."""

from __future__ import annotations

import numpy as np

_T0 = 293.15  # reference temperature, K
_T01 = 273.16  # triple point of water, K
_PR = 101.325  # reference pressure, kPa


def air_absorption_db_per_m(
    f, temperature_c: float = 20.0, humidity_pct: float = 50.0, pressure_kpa: float = _PR
) -> np.ndarray:
    """Atmospheric absorption coefficient in dB per metre (ISO 9613-1:1993).

    Absorption comes from viscosity/heat conduction (grows as f^2) and from
    the vibrational relaxation of O2 and N2 molecules, whose relaxation
    frequencies depend strongly on humidity. Example, 20 C and 50 % RH:
    about 0.005 dB/m at 1 kHz and 0.1 dB/m at 8 kHz.
    """
    f = np.asarray(f, dtype=float)
    T = temperature_c + 273.15
    pa = pressure_kpa / _PR
    tr = T / _T0
    C = -6.8346 * (_T01 / T) ** 1.261 + 4.6151
    h = humidity_pct * 10**C / pa  # molar concentration of water vapour, %
    fr_o = pa * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pa * tr**-0.5 * (9 + 280 * h * np.exp(-4.170 * (tr ** (-1 / 3) - 1)))
    return 8.686 * f**2 * (
        1.84e-11 / pa * tr**0.5
        + tr**-2.5
        * (
            0.01275 * np.exp(-2239.1 / T) / (fr_o + f**2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / T) / (fr_n + f**2 / fr_n)
        )
    )


def spreading_loss_db(distance_m: float, ref_m: float = 1.0) -> float:
    """Spherical spreading loss relative to ``ref_m`` (6 dB per doubling)."""
    return 20 * np.log10(distance_m / ref_m)
