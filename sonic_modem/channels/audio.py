"""Real speaker -> microphone channel through ``sounddevice``.

Not exercised yet (no hardware at hand); it implements the same
:class:`Channel` interface as the simulator so that ``Link.send`` works
unchanged once a speaker and microphone are available::

    from sonic_modem.channels.audio import SoundDeviceChannel
    ch = SoundDeviceChannel(fs=48_000)
    result = link.send(ch, bits)

``sounddevice`` is imported lazily so the rest of the package does not need
PortAudio installed. Install it with ``pip install sonic-modem[audio]``.
"""

from __future__ import annotations

import numpy as np

from .base import Channel


class SoundDeviceChannel(Channel):
    def __init__(
        self,
        fs: int = 48_000,
        lead: float = 0.3,
        tail: float = 0.5,
        input_device=None,
        output_device=None,
        volume: float = 0.5,
    ):
        self.fs = fs
        self.lead = lead
        self.tail = tail
        self.device = (input_device, output_device)
        self.volume = volume

    def transmit(self, samples: np.ndarray) -> np.ndarray:
        import sounddevice as sd  # lazy: optional dependency

        x = np.asarray(samples, dtype=np.float32) * self.volume
        pad = lambda s: np.zeros(int(round(s * self.fs)), np.float32)  # noqa: E731
        out = np.concatenate([pad(self.lead), x, pad(self.tail)])
        rec = sd.playrec(out, samplerate=self.fs, channels=1, device=self.device, blocking=True)
        return rec[:, 0].astype(np.float64)
