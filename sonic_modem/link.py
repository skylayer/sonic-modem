"""Frame assembly: preamble + gap + modulated payload.

``Link`` glues a :class:`~sonic_modem.modems.base.Modem` to a
:class:`~sonic_modem.sync.ChirpPreamble`. Swap the modem to compare schemes;
swap the channel (simulated or real audio) without touching either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channels.base import Channel
from .modems.base import Modem
from .sync import ChirpPreamble, Detection


@dataclass
class RxResult:
    bits: np.ndarray | None
    detection: Detection | None
    #: index in the recording where the first data symbol starts
    data_start: int | None

    @property
    def ok(self) -> bool:
        return self.bits is not None


class Link:
    def __init__(self, modem: Modem, preamble: ChirpPreamble | None = None, gap: float = 0.01):
        self.modem = modem
        self.preamble = preamble or ChirpPreamble(fs=modem.fs)
        if self.preamble.fs != modem.fs:
            raise ValueError("preamble and modem sample rates differ")
        #: silence between preamble and data, lets the chirp's echoes decay
        self.gap_samples = int(round(gap * modem.fs))

    @property
    def fs(self) -> int:
        return self.modem.fs

    @property
    def data_offset(self) -> int:
        return len(self.preamble) + self.gap_samples

    def build_frame(self, bits: np.ndarray) -> np.ndarray:
        data = self.modem.modulate(bits)
        return np.concatenate([self.preamble.waveform, np.zeros(self.gap_samples), data])

    def receive(self, rx: np.ndarray, n_bits: int) -> RxResult:
        det = self.preamble.detect(rx)
        if det is None:
            return RxResult(None, None, None)
        start = det.start + self.data_offset
        need = self.modem.n_samples(n_bits)
        if start + need > len(rx):
            return RxResult(None, det, start)
        bits = self.modem.demodulate(rx[start : start + need], n_bits)
        return RxResult(bits, det, start)

    def send(self, channel: Channel, bits: np.ndarray) -> RxResult:
        """Build a frame, push it through ``channel`` and decode it."""
        return self.receive(channel.transmit(self.build_frame(bits)), len(bits))
