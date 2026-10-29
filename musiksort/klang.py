"""Klang-Messungen für die Duplikat-Entscheidung (nur für Kandidaten, zwischengespeichert in der DB).

- Stille am Anfang und Ende: zwei Dateien derselben Aufnahme unterscheiden sich oft nur darin.
- Ende-Fingerabdruck auf das letzte Hörbare ausgerichtet (nicht aufs Dateiende): sonst passt das
  Ende bei unterschiedlich langer Stille nicht zusammen.
- Grenzfrequenz: verlustbehaftete Kodierer schneiden oberhalb ~16–20 kHz hart ab. Eine FLAC/WAV mit
  so einer Kante ist eine aufgeblasene MP3 („falsche FLAC“) und gewinnt nicht gegen eine echte 320er.
"""
import subprocess

import numpy as np

from . import config

ENDE_SEKUNDEN = 120
STILLE_DB = -50.0        # darunter gilt ein 50-ms-Fenster als still (dBFS)
STILLE_FENSTER = 0.05
RAND_SEKUNDEN = 30       # so viel wird am Anfang/Ende auf Stille untersucht
ECHT_AB_HZ = 19500       # echte verlustfreie Aufnahmen reichen meist bis 20–22 kHz
KANTE_DB = 25.0          # Abfall innerhalb von 500 Hz, ab dem eine Kodierer-Kante vorliegt


def _pcm(pfad, *vor, rate=8000, dauer=None):
    """Mono-PCM (float32) eines Ausschnitts; vor = ffmpeg-Optionen vor -i (z. B. -sseof)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", *vor, "-i", str(pfad), "-vn", "-ac", "1", "-ar", str(rate)]
    if dauer:
        cmd += ["-t", str(dauer)]
    r = subprocess.run(cmd + ["-f", "f32le", "-"], capture_output=True, timeout=300)
    if r.returncode:
        raise RuntimeError("nicht lesbar: " + r.stderr.decode(errors="replace")[:120])
    return np.frombuffer(r.stdout, np.float32)


def stille_laenge(a, rate, vom_ende=False):
    """Sekunden Stille am Anfang (bzw. Ende) eines Ausschnitts."""
    n = int(STILLE_FENSTER * rate)
    if a.size < n:
        return 0.0
    f = a[: a.size // n * n].reshape(-1, n)
    db = 20 * np.log10(np.sqrt((f.astype(np.float64) ** 2).mean(1)) + 1e-12)
    laut = np.flatnonzero(db > STILLE_DB)
    if laut.size == 0:
        return round(f.shape[0] * STILLE_FENSTER, 2)  # ganz still
    k = (f.shape[0] - 1 - laut[-1]) if vom_ende else laut[0]
    return round(k * STILLE_FENSTER, 2)


def stille(con, m):
    """(Stille am Anfang, Stille am Ende) in Sekunden; zwischengespeichert."""
    if m.get("stille_anfang") is not None and m.get("stille_ende") is not None:
        return m["stille_anfang"], m["stille_ende"]
    pfad = config.medienpfad(m["pfad"])
    anfang = stille_laenge(_pcm(pfad, dauer=RAND_SEKUNDEN), 8000)
    ende = stille_laenge(_pcm(pfad, "-sseof", f"-{RAND_SEKUNDEN}"), 8000, vom_ende=True)
    m["stille_anfang"], m["stille_ende"] = anfang, ende
    with con:
        con.execute("UPDATE lieder SET stille_anfang=?, stille_ende=? WHERE id=?", (anfang, ende, m["id"]))
    return anfang, ende


def netto_dauer(m):
    """Länge ohne Stille am Anfang und Ende (nur wenn gemessen, sonst die Dateilänge)."""
    d = m.get("dauer") or 0
    if m.get("stille_anfang") is None or m.get("stille_ende") is None:
        return d
    return max(0.0, d - m["stille_anfang"] - m["stille_ende"])


def fp_ende(con, m):
    """Chromaprint der letzten 2 Minuten bis zum letzten Hörbaren (Stille am Ende abgezogen)."""
    if m.get("fp_ende2"):
        return np.frombuffer(m["fp_ende2"], np.uint32)
    import acoustid
    import chromaprint
    _, ende = stille(con, m)
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-sseof", f"-{ENDE_SEKUNDEN + ende:.2f}",
                        "-i", str(config.medienpfad(m["pfad"])), "-t", str(ENDE_SEKUNDEN), "-vn", "-ac", "1",
                        "-ar", str(config.SR), "-f", "s16le", "-"], capture_output=True, timeout=300)
    if r.returncode or len(r.stdout) < config.SR * 2 * 10:
        raise RuntimeError("Ende nicht lesbar")
    roh, _ = chromaprint.decode_fingerprint(
        acoustid.fingerprint(config.SR, 1, iter([r.stdout]), maxlength=ENDE_SEKUNDEN))
    fp = np.asarray(roh, dtype=np.int64).astype(np.uint32)
    m["fp_ende2"] = fp.tobytes()
    with con:
        con.execute("UPDATE lieder SET fp_ende2=? WHERE id=?", (m["fp_ende2"], m["id"]))
    return fp


def grenze_messen(a, rate):
    """(Grenzfrequenz in Hz, Abfall in dB an der Grenze) aus einem Ausschnitt in voller Abtastrate."""
    n = 4096
    if a.size < n * 4:
        raise RuntimeError("zu kurz für die Frequenzmessung")
    fenster = np.hanning(n)
    schritte = range(0, a.size - n, n // 2)
    p = np.zeros(n // 2 + 1)
    for s in schritte:
        p += np.abs(np.fft.rfft(a[s:s + n] * fenster)) ** 2
    db = 10 * np.log10(p / len(schritte) + 1e-20)
    hz = np.fft.rfftfreq(n, 1 / rate)
    glatt = np.convolve(db, np.ones(9) / 9, mode="same")
    bezug = np.median(glatt[(hz >= 2000) & (hz <= 10000)])
    ueber = np.flatnonzero((glatt > bezug - 45) & (hz < rate / 2 - 500))
    if ueber.size == 0:
        return 0.0, 0.0
    i = int(ueber[-1])
    breite = int(round(500 / (hz[1] - hz[0])))
    davor = glatt[max(0, i - breite): i + 1].max()
    danach = glatt[i + 1: i + 1 + breite].mean() if i + 1 < glatt.size else davor
    return float(round(hz[i])), float(round(davor - danach, 1))


def grenzfrequenz(con, m):
    """(Grenzfrequenz, Kante in dB) eines 30-s-Ausschnitts aus der Mitte; zwischengespeichert."""
    if m.get("grenzfrequenz") is not None:
        return m["grenzfrequenz"], m.get("kante") or 0.0
    rate = max(44100, int(m.get("samplerate") or 44100))
    mitte = max(0.0, (m.get("dauer") or 60) / 2 - 15)
    a = _pcm(config.medienpfad(m["pfad"]), "-ss", f"{mitte:.1f}", rate=rate, dauer=30)
    gf, kante = grenze_messen(a, rate)
    m["grenzfrequenz"], m["kante"] = gf, kante
    with con:
        con.execute("UPDATE lieder SET grenzfrequenz=?, kante=? WHERE id=?", (gf, kante, m["id"]))
    return gf, kante


def falsche_verlustfrei(r):
    """True, wenn eine „verlustfreie“ Datei laut Messung eine Kodierer-Kante unter ~19,5 kHz hat."""
    gf, kante = _feld(r, "grenzfrequenz"), _feld(r, "kante")
    return bool(_feld(r, "verlustfrei")) and gf is not None and 0 < gf < ECHT_AB_HZ and (kante or 0) >= KANTE_DB


def geschaetzte_bitrate(gf):
    """Grobe MP3-Bitrate aus der Grenzfrequenz (LAME-Standard-Tiefpässe)."""
    for grenze, kbit in ((15500, 96), (16500, 128), (17500, 160), (19000, 192), (19700, 256)):
        if gf < grenze:
            return kbit
    return 320


def _feld(r, k):
    try:
        return r[k]
    except (KeyError, IndexError):
        return None
