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
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=d=40", str(m / "still.wav")], check=True)
    # Handyclip: Bild + Ton; daneben ein Lied mit eingebettetem Cover (bleibt Musik)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=20:s=320x240", "-i", str(m / "beat.mp3"),
                    "-t", "20", "-c:v", "libx264", "-c:a", "aac", str(m / "clip.mp4")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=64x64:d=1", "-frames:v", "1",
                    str(tmp_path / "cover.png")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(m / "beat.mp3"), "-i", str(tmp_path / "cover.png"),
                    "-map", "0:a", "-map", "1:v", "-c", "copy", "-id3v2_version", "3",
                    "-disposition:v", "attached_pic", str(m / "cover.mp3")], check=True)
    (m / "._beat.mp3").write_bytes(b"macOS-Hilfsdatei")
    return tmp_path / "media"


def test_suchen_analysieren_fehlt(media):
    scan.suchen()
    con = db.verbinden()
    pfade = sorted(r[0] for r in con.execute("SELECT pfad FROM lieder"))
    assert pfade == ["Musik/beat.flac", "Musik/beat.mp3", "Musik/clip.mp4", "Musik/cover.mp3",
                     "Musik/kaputt.mp3", "Musik/still.wav"]   # ._ übersprungen

    scan.analysieren(jobs=1)
    st = dict(con.execute("SELECT pfad, status FROM lieder").fetchall())
    assert st == {"Musik/beat.flac": "analysiert", "Musik/beat.mp3": "analysiert", "Musik/clip.mp4": "video",
                  "Musik/cover.mp3": "analysiert", "Musik/kaputt.mp3": "fehler", "Musik/still.wav": "stumm"}
    # schon erfasste Videos nachträglich aussortieren; ein Video bleibt beim nächsten Suchen „video“
    con.execute("UPDATE lieder SET status='analysiert' WHERE pfad='Musik/clip.mp4'")
    con.commit()
    assert len(scan.videos_aussortieren(ausgabe=lambda *_: None)) == 1
    still, beat = (con.execute("SELECT id FROM lieder WHERE pfad=?", (p,)).fetchone()[0]
                   for p in ("Musik/still.wav", "Musik/beat.mp3"))
    con.execute("UPDATE lieder SET status='analysiert' WHERE id=?", (still,))
    con.commit()
    assert scan.stumme_aussortieren([still, beat], ausgabe=lambda *_: None) == [still]
    scan.suchen()
    assert con.execute("SELECT status FROM lieder WHERE pfad='Musik/clip.mp4'").fetchone()[0] == "video"
    r = con.execute("SELECT bpm, snippet_start, length(emb), versuche FROM lieder WHERE pfad='Musik/beat.mp3'").fetchone()
    assert 110 <= r[0] <= 130 and r[1] is not None and r[2] == 1280 * 2 and r[3] == 0
    # alter Bestand (anderes Tempo-Modell) wird nachgemessen, nur bpm + tempo_stand ändern sich
    con.execute("UPDATE lieder SET bpm=60, tempo_stand=NULL WHERE pfad='Musik/beat.mp3'")
    con.commit()
    assert scan.tempo_nachmessen(jobs=1, ausgabe=lambda *_: None) == 1
    bpm, stand = con.execute("SELECT bpm, tempo_stand FROM lieder WHERE pfad='Musik/beat.mp3'").fetchone()
    assert 110 <= bpm <= 130 and stand == config.TEMPO_STAND
    assert scan.tempo_nachmessen(jobs=1, ausgabe=lambda *_: None) == 0

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
