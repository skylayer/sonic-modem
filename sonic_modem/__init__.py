"""sonic-modem: digital communication over sound."""

from .channels import AcousticChannelConfig, Channel, IdealChannel, SimulatedAcousticChannel
from .link import Link, RxResult
from .modems import FSKModem, Modem
from .sync import ChirpPreamble

__all__ = [
    "AcousticChannelConfig",
    "Channel",
    "ChirpPreamble",
    "FSKModem",
    "IdealChannel",
    "Link",
    "Modem",
    "RxResult",
    "SimulatedAcousticChannel",
]
