"""A simulated speaker -> room -> microphone channel.

The signal path is::

    tx -> speaker band-pass -> room impulse response (direct path + echoes)
       -> air absorption -> attenuation -> microphone band-pass -> clock offset/drift
       -> unknown start delay -> + AWGN -> rx

Each stage can be switched off through :class:`AcousticChannelConfig`, which
makes it easy to see what each impairment costs a given modem.

SNR convention: ``snr_db`` is the ratio of the *received* signal power
(averaged over the part of the recording that carries signal) to the total
noise power over the full band 0..fs/2. Helpers in :mod:`sonic_modem.metrics`
convert it to Eb/N0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from ..acoustics import air_absorption_db_per_m
from .base import Channel


@dataclass
class AcousticChannelConfig:
    fs: int = 48_000
    #: speaker pass band (Hz); None disables the filter. Small speakers roll
    #: off hard below a few hundred Hz.
    speaker_band: tuple[float, float] | None = (300.0, 8_000.0)
    #: microphone pass band (Hz); None disables the filter.
    mic_band: tuple[float, float] | None = (100.0, 12_000.0)
    filter_order: int = 4
    #: room echoes as (delay seconds, relative gain) on top of the direct path
    echoes: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.003, 0.4), (0.009, -0.25), (0.021, 0.12)]
    )
    #: speaker-to-mic distance for frequency-dependent air absorption
    #: (ISO 9613-1); None disables it. Broadband losses such as spherical
    #: spreading belong in ``attenuation_db``.
    distance_m: float | None = 1.0
    temperature_c: float = 20.0
    humidity_pct: float = 50.0
    #: overall broadband loss from speaker to microphone, in dB (positive = quieter)
    attenuation_db: float = 20.0
    #: random silence before the signal starts, as (min, max) seconds
    start_delay: tuple[float, float] = (0.05, 0.5)
    #: silence recorded after the signal ends, in seconds
    tail: float = 0.1
    #: fractional sample offset is drawn from U(0, 1) when True
    fractional_delay: bool = True
    #: sample-clock mismatch between the two sound cards, parts per million
    clock_ppm: float = 0.0
    #: signal-to-noise ratio in dB (see module docstring); None = no noise
    snr_db: float | None = 20.0


class SimulatedAcousticChannel(Channel):
    def __init__(self, config: AcousticChannelConfig | None = None, seed: int | None = None):
        self.config = config or AcousticChannelConfig()
        self.fs = self.config.fs
        self.rng = np.random.default_rng(seed)
        c = self.config
        self._speaker = self._bandpass(c.speaker_band) if c.speaker_band else None
        self._mic = self._bandpass(c.mic_band) if c.mic_band else None
        self.impulse_response = self._room_ir()
        #: details of the last transmission (true delay, noise power, ...)
        self.last: dict = {}

    # -------------------------------------------------------------- stages
    def _bandpass(self, band: tuple[float, float]):
        lo, hi = band
        return signal.butter(self.config.filter_order, [lo, hi], "bandpass", fs=self.fs, output="sos")

    def _room_ir(self) -> np.ndarray:
        echoes = self.config.echoes or []
        n = 1 + max([int(round(d * self.fs)) for d, _ in echoes], default=0)
        h = np.zeros(n)
        h[0] = 1.0
        for d, g in echoes:
            h[int(round(d * self.fs))] += g
        return h

    def air_loss_db(self, f) -> np.ndarray:
        c = self.config
        if c.distance_m is None:
            return np.zeros_like(np.asarray(f, dtype=float))
        return c.distance_m * air_absorption_db_per_m(f, c.temperature_c, c.humidity_pct)

    def _air(self, x: np.ndarray) -> np.ndarray:
        """Zero-phase frequency-domain air absorption filter."""
        if self.config.distance_m is None or len(x) == 0:
            return x
        n = len(x)
        gain = 10 ** (-self.air_loss_db(np.fft.rfftfreq(n, 1 / self.fs)) / 20)
        return np.fft.irfft(np.fft.rfft(x) * gain, n)

    def _resample(self, x: np.ndarray, frac: float) -> np.ndarray:
        """Apply a fractional delay and clock drift by band-limited interpolation."""
        ppm = self.config.clock_ppm
        if frac == 0 and ppm == 0:
            return x
        ratio = 1.0 + ppm * 1e-6  # rx clock ticks per tx clock tick
        n_out = int(np.floor((len(x) - 1) / ratio)) + 1
        t = np.arange(n_out) * ratio - frac  # where each rx sample falls in tx time
        # windowed-sinc interpolation, 16 taps per side
        half = 16
        base = np.floor(t).astype(np.int64)
        mu = t - base
        taps = np.arange(-half + 1, half + 1)
        idx = base[:, None] + taps[None, :]
        d = mu[:, None] - taps[None, :]
        w = np.sinc(d) * np.kaiser(2 * half, 6.0)[None, :]
        xp = np.concatenate([np.zeros(half), x, np.zeros(half)])
        return (xp[np.clip(idx + half, 0, len(xp) - 1)] * w).sum(axis=1)

    # -------------------------------------------------------------- public
    def transmit(self, samples: np.ndarray) -> np.ndarray:
        c = self.config
        x = np.asarray(samples, dtype=np.float64)
        # let filter and echo tails ring out
        x = np.concatenate([x, np.zeros(len(self.impulse_response) + int(0.01 * self.fs))])
        if self._speaker is not None:
            x = signal.sosfilt(self._speaker, x)
        x = signal.fftconvolve(x, self.impulse_response)[: len(x)]
        x = self._air(x)
        x *= 10 ** (-c.attenuation_db / 20)
        if self._mic is not None:
            x = signal.sosfilt(self._mic, x)

        frac = self.rng.uniform(0, 1) if c.fractional_delay else 0.0
        x = self._resample(x, frac)
        lo, hi = c.start_delay
        delay = int(round(self.rng.uniform(lo, hi) * self.fs))

        signal_power = float(np.mean(x[: len(samples)] ** 2)) if len(samples) else 0.0
        y = np.concatenate([np.zeros(delay), x, np.zeros(int(round(c.tail * self.fs)))])
        noise_power = 0.0
        if c.snr_db is not None and signal_power > 0:
            noise_power = signal_power / 10 ** (c.snr_db / 10)
            y = y + self.rng.normal(0.0, np.sqrt(noise_power), len(y))

        self.last = {
            "delay_samples": delay + frac,
            "signal_power": signal_power,
            "noise_power": noise_power,
        }
        return y

    def frequency_response(self, n: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Magnitude response (dB) of the linear part of the channel."""
        f = np.fft.rfftfreq(2 * n, 1 / self.fs)
        H = np.fft.rfft(self.impulse_response, 2 * n) * 10 ** (-self.config.attenuation_db / 20)
        H = H * 10 ** (-self.air_loss_db(f) / 20)
        for sos in (self._speaker, self._mic):
            if sos is not None:
                _, h = signal.sosfreqz(sos, worN=f, fs=self.fs)
                H = H * h
        return f, 20 * np.log10(np.abs(H) + 1e-12)
