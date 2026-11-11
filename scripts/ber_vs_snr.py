#!/usr/bin/env python3
"""Sweep Eb/N0 and measure end-to-end BER through the simulated channel.

Every frame goes through the full receive chain: chirp preamble detection,
alignment, demodulation. A frame whose preamble is missed counts every bit as
an error.

    python scripts/ber_vs_snr.py                # BFSK + 4-FSK, AWGN and room
    python scripts/ber_vs_snr.py --M 2 --bits 50000 --out results/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from sonic_modem import AcousticChannelConfig, FSKModem, Link, SimulatedAcousticChannel
from sonic_modem.bits import random_bits
from sonic_modem.metrics import ber_noncoherent_mfsk, bit_errors, ebn0_to_snr_db

CHANNELS = {
    # white noise only, still with unknown start delay so sync is exercised
    "awgn": dict(speaker_band=None, mic_band=None, echoes=[], attenuation_db=0.0),
    # band-limited speaker/mic, three echoes, 20 dB loss, 30 ppm clock offset
    "room": dict(clock_ppm=30.0),
}


def measure(modem, channel_kw, ebn0_db, n_bits, frame_bits, max_errors, seed):
    rng = np.random.default_rng(seed)
    link = Link(modem)
    snr = float(ebn0_to_snr_db(ebn0_db, modem.fs, modem.bit_rate))
    ch = SimulatedAcousticChannel(AcousticChannelConfig(snr_db=snr, **channel_kw), seed=seed)
    errors = total = missed = 0
    while total < n_bits and errors < max_errors:
        bits = random_bits(frame_bits, rng)
        res = link.send(ch, bits)
        missed += not res.ok
        errors += bit_errors(bits, res.bits)
        total += frame_bits
    return errors, total, missed, snr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=int, nargs="+", default=[2, 4], help="FSK orders to test")
    ap.add_argument("--baud", type=float, default=100.0)
    ap.add_argument("--ebn0", type=float, nargs=3, default=[0.0, 14.0, 1.0], metavar=("START", "STOP", "STEP"))
    ap.add_argument("--bits", type=int, default=40_000, help="max bits per point")
    ap.add_argument("--frame-bits", type=int, default=1_000)
    ap.add_argument("--max-errors", type=int, default=400, help="stop a point early after this many errors")
    ap.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    start, stop, step = args.ebn0
    grid = np.arange(start, stop + step / 2, step)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for M in args.M:
        modem = FSKModem(M=M, baud=args.baud)
        print(modem.describe())
        for name in args.channels:
            for i, e in enumerate(grid):
                err, tot, missed, snr = measure(
                    modem, CHANNELS[name], e, args.bits, args.frame_bits, args.max_errors, args.seed + i
                )
                ber = err / tot
                theory = float(ber_noncoherent_mfsk(e, M))
                rows.append(dict(M=M, channel=name, ebn0_db=e, snr_db=snr, bits=tot, errors=err,
                                 missed_frames=missed, ber=ber, theory=theory, bit_rate=modem.bit_rate))
                print(f"  {name:5s} Eb/N0={e:5.1f} dB  SNR={snr:6.1f} dB  BER={ber:.2e}  "
                      f"(theory {theory:.2e}, {err}/{tot}, missed {missed})")

    csv_path = args.out / "ber_fsk.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    if not args.no_plot:
        plot(rows, grid, args.out / "ber_fsk.png")


def plot(rows, grid, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fine = np.linspace(grid[0], grid[-1], 200)
    styles = {"awgn": "o-", "room": "s--"}
    for i, M in enumerate(sorted({r["M"] for r in rows})):
        color = f"C{i}"
        ax.semilogy(fine, ber_noncoherent_mfsk(fine, M), ":", color=color, lw=1,
                    label=f"{M}-FSK theory (AWGN)")
        for name in dict.fromkeys(r["channel"] for r in rows):
            pts = [r for r in rows if r["M"] == M and r["channel"] == name and r["errors"] > 0]
            if pts:
                rate = pts[0]["bit_rate"]
                ax.semilogy([r["ebn0_db"] for r in pts], [r["ber"] for r in pts], styles[name],
                            color=color, ms=4, label=f"{M}-FSK {name} ({rate:.0f} bit/s)")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("bit error rate")
    ax.set_ylim(1e-5, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Non-coherent FSK through the simulated acoustic channel")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
