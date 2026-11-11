import numpy as np
import pytest

from sonic_modem import (
    AcousticChannelConfig,
    FSKModem,
    IdealChannel,
    Link,
    SimulatedAcousticChannel,
)
from sonic_modem.bits import gray_decode, gray_encode, random_bits
from sonic_modem.metrics import ber_noncoherent_mfsk, bit_errors


def test_gray_roundtrip():
    n = np.arange(64)
    assert np.array_equal(gray_decode(gray_encode(n)), n)
    # neighbours differ in exactly one bit
    g = gray_encode(n)
    assert all(bin(a ^ b).count("1") == 1 for a, b in zip(g[:-1], g[1:]))


@pytest.mark.parametrize("M", [2, 4, 8])
def test_fsk_loopback(M):
    modem = FSKModem(M=M, center_freq=3000)
    bits = random_bits(301, np.random.default_rng(0))
    assert np.array_equal(modem.demodulate(modem.modulate(bits), len(bits)), bits)


def test_fsk_rejects_bad_params():
    with pytest.raises(ValueError):
        FSKModem(M=3)
    with pytest.raises(ValueError):
        FSKModem(baud=97)


def test_link_ideal_channel():
    link = Link(FSKModem())
    bits = random_bits(200, np.random.default_rng(1))
    rx = np.concatenate([np.zeros(1234), IdealChannel().transmit(link.build_frame(bits))])
    res = link.receive(rx, len(bits))
    assert res.ok and res.detection.start == 1234
    assert np.array_equal(res.bits, bits)


def test_preamble_not_found_in_noise():
    link = Link(FSKModem())
    res = link.receive(np.random.default_rng(2).normal(0, 0.1, 48_000), 100)
    assert not res.ok and res.detection is None


def test_sync_finds_true_delay_through_echoes():
    ch = SimulatedAcousticChannel(AcousticChannelConfig(snr_db=10), seed=3)
    link = Link(FSKModem())
    for _ in range(5):
        res = link.send(ch, random_bits(100))
        assert res.detection is not None
        # band-pass filters add a few samples of group delay
        assert 0 <= res.detection.start - ch.last["delay_samples"] < 30


def test_channel_snr_is_as_configured():
    ch = SimulatedAcousticChannel(
        AcousticChannelConfig(snr_db=10, start_delay=(0, 0), tail=0), seed=4
    )
    x = FSKModem().modulate(random_bits(2000, np.random.default_rng(4)))
    ch.transmit(x)
    measured = 10 * np.log10(ch.last["signal_power"] / ch.last["noise_power"])
    assert measured == pytest.approx(10, abs=1e-9)


def test_fsk_error_free_at_high_snr_through_room():
    ch = SimulatedAcousticChannel(AcousticChannelConfig(snr_db=15, clock_ppm=50), seed=5)
    link = Link(FSKModem(M=4))
    bits = random_bits(2000, np.random.default_rng(5))
    assert bit_errors(bits, link.send(ch, bits).bits) == 0


def test_awgn_ber_matches_theory():
    """Pure AWGN (no filters/echo): measured BFSK BER is close to theory."""
    modem = FSKModem(guard=0.0)
    bits = random_bits(40_000, np.random.default_rng(6))
    x = modem.modulate(bits)
    ebn0_db = 8.0
    p = np.mean(x**2)
    snr = ebn0_db - 10 * np.log10(modem.fs / (2 * modem.bit_rate))
    rng = np.random.default_rng(7)
    y = x + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(x))
    ber = bit_errors(bits, modem.demodulate(y, len(bits))) / len(bits)
    theory = float(ber_noncoherent_mfsk(ebn0_db, 2))
    assert ber == pytest.approx(theory, rel=0.25)


def test_air_absorption_matches_iso_9613_table():
    from sonic_modem.acoustics import air_absorption_db_per_m

    # ISO 9613-1 Table 1, 20 C, 50 % RH, 101.325 kPa (dB/km)
    for f, ref in [(1000, 4.66), (2000, 9.86), (4000, 29.7), (8000, 105)]:
        assert air_absorption_db_per_m(f) * 1000 == pytest.approx(ref, rel=0.02)


def test_channel_applies_air_absorption():
    cfg = dict(speaker_band=None, mic_band=None, echoes=[], attenuation_db=0,
               start_delay=(0, 0), tail=0, fractional_delay=False, snr_db=None)
    fs = 48_000
    t = np.arange(fs) / fs
    tone = np.sin(2 * np.pi * 12_000 * t)
    ch = SimulatedAcousticChannel(AcousticChannelConfig(distance_m=50, **cfg))
    y = ch.transmit(tone)[: fs]
    loss = 10 * np.log10(np.mean(tone[1000:-1000] ** 2) / np.mean(y[1000:-1000] ** 2))
    assert loss == pytest.approx(float(ch.air_loss_db(12_000)), abs=0.1)
