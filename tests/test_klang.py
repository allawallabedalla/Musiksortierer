"""Klang-Messungen: Stille am Rand, Grenzfrequenz/Kodierer-Kante."""
import numpy as np

from musiksort import klang


def test_stille_am_anfang_und_ende():
    rate = 8000
    ton = 0.5 * np.sin(2 * np.pi * 440 * np.arange(rate * 5) / rate).astype(np.float32)
    a = np.concatenate([np.zeros(rate * 2, np.float32), ton, np.zeros(int(rate * 1.5), np.float32)])
    assert abs(klang.stille_laenge(a, rate) - 2.0) <= 0.05
    assert abs(klang.stille_laenge(a, rate, vom_ende=True) - 1.5) <= 0.05


def test_grenzfrequenz_erkennt_kante_und_volles_spektrum():
    rate = 44100
    rng = np.random.default_rng(0)
    voll = rng.standard_normal(rate * 10).astype(np.float32) * 0.1
    spek = np.fft.rfft(voll)
    spek[np.fft.rfftfreq(voll.size, 1 / rate) > 16000] = 0   # harter Tiefpass wie ein MP3-Kodierer
    kante = np.fft.irfft(spek, voll.size).astype(np.float32)
    gf, db = klang.grenze_messen(kante, rate)
    assert 15500 < gf < 16500 and db >= klang.KANTE_DB
    gf2, _ = klang.grenze_messen(voll, rate)
    assert gf2 > klang.ECHT_AB_HZ


def test_falsche_verlustfrei_und_bitrate():
    assert klang.falsche_verlustfrei({"verlustfrei": 1, "grenzfrequenz": 16000.0, "kante": 40.0})
    assert not klang.falsche_verlustfrei({"verlustfrei": 1, "grenzfrequenz": 21000.0, "kante": 40.0})
    assert not klang.falsche_verlustfrei({"verlustfrei": 1, "grenzfrequenz": 16000.0, "kante": 5.0})  # weicher Abfall: alte Aufnahme
    assert not klang.falsche_verlustfrei({"verlustfrei": 0, "grenzfrequenz": 16000.0, "kante": 40.0})
    assert klang.geschaetzte_bitrate(16000) == 128 and klang.geschaetzte_bitrate(20500) == 320
