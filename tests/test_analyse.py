"""Ende-zu-Ende: suchen + analysieren mit echten Modellen auf künstlichen Dateien (dauert ~30 s)."""
import subprocess

import pytest

from musiksort import config, db, scan


@pytest.fixture
def media(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PFAD", tmp_path / "musik.db")
    monkeypatch.setattr(config, "MEDIA_ROOT", tmp_path / "media")
    monkeypatch.setattr(config, "MOUNT_PRUEFEN", False)
    monkeypatch.setattr(config, "EINSTELLUNGEN_DA", True)
    m = tmp_path / "media" / "Musik"
    m.mkdir(parents=True)
    beat = "aevalsrc='0.6*sin(2*PI*55*t)*exp(-12*mod(t\\,0.5))':d=200"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", beat, "-b:a", "192k", str(m / "beat.mp3")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(m / "beat.mp3"), str(m / "beat.flac")], check=True)
    (m / "kaputt.mp3").write_bytes(b"\x00\xff" * 5000)
    (m / "._beat.mp3").write_bytes(b"macOS-Hilfsdatei")
    return tmp_path / "media"


def test_suchen_analysieren_fehlt(media):
    scan.suchen()
    con = db.verbinden()
    pfade = sorted(r[0] for r in con.execute("SELECT pfad FROM lieder"))
    assert pfade == ["Musik/beat.flac", "Musik/beat.mp3", "Musik/kaputt.mp3"]   # ._ übersprungen

    scan.analysieren(jobs=1)
    st = dict(con.execute("SELECT pfad, status FROM lieder").fetchall())
    assert st == {"Musik/beat.flac": "analysiert", "Musik/beat.mp3": "analysiert", "Musik/kaputt.mp3": "fehler"}
    r = con.execute("SELECT bpm, snippet_start, length(emb), versuche FROM lieder WHERE pfad='Musik/beat.mp3'").fetchone()
    assert 110 <= r[0] <= 130 and r[1] is not None and r[2] == 1280 * 2 and r[3] == 0

    (media / "Musik" / "beat.flac").unlink()
    scan.suchen()
    assert con.execute("SELECT status FROM lieder WHERE pfad='Musik/beat.flac'").fetchone()[0] == "fehlt"


def test_ohne_nas_kein_fehler_markieren(media, monkeypatch):
    scan.suchen()
    monkeypatch.setattr(config, "MOUNT_PRUEFEN", True)   # tmp ist kein Mount: gilt als „NAS weg“
    with pytest.raises(SystemExit):
        scan.analysieren(jobs=1)
    con = db.verbinden()
    assert {r[0] for r in con.execute("SELECT status FROM lieder")} == {"neu"}


def test_ohne_einstellungen_keine_suche_in_bestehender_sammlung(media, monkeypatch):
    scan.suchen()                                   # erste Suche: Sammlung entsteht
    monkeypatch.setattr(config, "EINSTELLUNGEN_DA", False)
    with pytest.raises(SystemExit, match="fehlt"):
        scan.suchen()
