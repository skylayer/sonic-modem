#!/usr/bin/env python3
"""Spectral views of the link.

Writes three figures:

* ``psd_fsk.png``: Welch PSD of the transmitted frame, the received signal,
  the noise and the full recording, with the FSK tones marked.
* ``air_attenuation.png``: atmospheric absorption (ISO 9613-1) versus
  frequency for several humidities, and the total air loss over distance.
* ``channel_response.png``: magnitude response of each stage of the simulated
  channel (speaker, mic, room echoes, air) and of the whole channel.

    python scripts/plot_spectra.py --out docs/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402

from sonic_modem import AcousticChannelConfig, FSKModem, Link, SimulatedAcousticChannel  # noqa: E402
from sonic_modem.acoustics import air_absorption_db_per_m  # noqa: E402
from sonic_modem.bits import random_bits  # noqa: E402
from sonic_modem.metrics import ebn0_to_snr_db  # noqa: E402


def psd_db(x, fs, nperseg=4096):
    f, p = signal.welch(x, fs, nperseg=nperseg)
    return f, 10 * np.log10(p + 1e-30)


def plot_psd(out: Path, M: int, ebn0_db: float, seed: int):
    modem = FSKModem(M=M)
    link = Link(modem)
    fs = modem.fs
    snr = float(ebn0_to_snr_db(ebn0_db, fs, modem.bit_rate))
    cfg = AcousticChannelConfig(snr_db=None, start_delay=(0.0, 0.0), tail=0.0)
    ch = SimulatedAcousticChannel(cfg, seed=seed)

    tx = link.build_frame(random_bits(2000, np.random.default_rng(seed)))
    rx_sig = ch.transmit(tx)
    p_sig = ch.last["signal_power"]
    noise = np.random.default_rng(seed + 1).normal(0, np.sqrt(p_sig / 10 ** (snr / 10)), len(rx_sig))
    rx = rx_sig + noise

    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    for ax, (lo, hi) in zip(axes, [(0, fs / 2), (500, 7_000)]):
        for x, label, style in [
            (tx, "TX frame (preamble + FSK)", dict(color="C0", lw=1)),
            (rx_sig, "RX signal (after channel)", dict(color="C1", lw=1)),
            (rx, "RX recording = signal + noise", dict(color="C3", lw=0.8, alpha=0.7)),
            (noise, f"noise (SNR {snr:.1f} dB full band)", dict(color="k", lw=1, ls="--")),
        ]:
            f, p = psd_db(x, fs)
            ax.plot(f, p, label=label, **style)
        for tone in modem.tones:
            ax.axvline(tone, color="k", ls=":", lw=0.7)
        ax.axvspan(link.preamble.f0, link.preamble.f1, color="C2", alpha=0.08, label="chirp band")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-160, -20)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (dB/Hz)")
        ax.grid(alpha=0.3)
    axes[0].set_title(f"{M}-FSK, {modem.bit_rate:.0f} bit/s, Eb/N0 = {ebn0_db:g} dB, Welch PSD")
    axes[1].set_title("zoom on the occupied band (dotted lines: FSK tones)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "psd_fsk.png", dpi=130)
    plt.close(fig)


def plot_air(out: Path):
    f = np.logspace(np.log10(50), np.log10(24_000), 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for rh in [10, 30, 50, 80]:
        a1.loglog(f, air_absorption_db_per_m(f, 20, rh), label=f"{rh} % RH")
    a1.set_title("Air absorption at 20 °C (ISO 9613-1)")
    a1.set_xlabel("frequency (Hz)")
    a1.set_ylabel("absorption (dB/m)")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend(fontsize=8)

    for d in [1, 10, 100]:
        a2.semilogx(f, d * air_absorption_db_per_m(f, 20, 50), label=f"{d} m")
    a2.axvspan(1_000, 6_000, color="C2", alpha=0.1, label="chirp preamble")
    a2.axvspan(2_000, 3_000, color="C3", alpha=0.15, label="FSK tones")
    a2.set_title("Total air loss, 20 °C, 50 % RH")
    a2.set_xlabel("frequency (Hz)")
    a2.set_ylabel("loss (dB)")
    a2.set_ylim(0, 40)
    a2.grid(True, which="both", alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "air_attenuation.png", dpi=130)
    plt.close(fig)


def plot_channel(out: Path, modem: FSKModem):
    base = AcousticChannelConfig()
    stages = {
        "speaker": dict(mic_band=None, echoes=[], distance_m=None),
        "mic": dict(speaker_band=None, echoes=[], distance_m=None),
        "room echoes": dict(speaker_band=None, mic_band=None, distance_m=None),
        "air, 1 m": dict(speaker_band=None, mic_band=None, echoes=[]),
    }
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, (name, kw) in enumerate(stages.items()):
        cfg = AcousticChannelConfig(**{**base.__dict__, **kw, "attenuation_db": 0.0})
        f, h = SimulatedAcousticChannel(cfg).frequency_response()
        ax.plot(f, h, lw=1, color=f"C{i}", alpha=0.8, label=name)
    cfg = AcousticChannelConfig(**{**base.__dict__, "attenuation_db": 0.0})
    f, h = SimulatedAcousticChannel(cfg).frequency_response()
    ax.plot(f, h, color="k", lw=1.4, label="whole channel")
    for tone in modem.tones:
        ax.axvline(tone, color="C3", ls=":", lw=0.8)
    ax.set_xlim(0, 12_000)
    ax.set_ylim(-40, 10)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("gain (dB, excluding broadband attenuation)")
    ax.set_title("Simulated channel response (dotted: 4-FSK tones)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "channel_response.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--ebn0", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_psd(args.out, args.M, args.ebn0, args.seed)
    plot_air(args.out)
    plot_channel(args.out, FSKModem(M=args.M))
    print(f"wrote psd_fsk.png, air_attenuation.png, channel_response.png to {args.out}")


if __name__ == "__main__":
    main()
