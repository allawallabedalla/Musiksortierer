"""Favoriten-Stern: in der App setzen, in die Datei schreiben, bei Duplikaten mitnehmen."""
import os
import subprocess

import mutagen
import pytest
from fastapi.testclient import TestClient

from musiksort import config, db, lernen, papierkorb, tags, web


def _datei(ziel, endung):
    ziel = ziel.with_suffix(endung)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3",
                    *(["-c:a", "aac"] if endung == ".m4a" else []), str(ziel)], check=True)
    return ziel


@pytest.mark.parametrize("endung", [".mp3", ".flac", ".m4a", ".ogg"])
def test_stern_setzen_und_entfernen_je_format(tmp_path, endung):
    pfad = _datei(tmp_path / "lied", endung)
    assert not tags.lesen(pfad)
    tags.schreiben(pfad, True)
    assert tags.lesen(pfad)
    if endung == ".mp3":
        popm = mutagen.File(pfad).tags.getall("POPM")
        assert popm and popm[0].rating == 255
    if endung in (".flac", ".ogg"):
        assert mutagen.File(pfad).tags["RATING"] == ["5"]
    tags.schreiben(pfad, False)
    assert not tags.lesen(pfad)


def test_bestehende_gruppierung_bleibt(tmp_path):
    pfad = _datei(tmp_path / "lied", ".flac")
    f = mutagen.File(pfad)
    f.tags["GROUPING"] = "Sommer 2019"
    f.save()
    tags.schreiben(pfad, True)
    assert mutagen.File(pfad).tags["GROUPING"] == ["Sommer 2019 / Favorit"]
    tags.schreiben(pfad, False)
    assert mutagen.File(pfad).tags["GROUPING"] == ["Sommer 2019"]


@pytest.fixture
def umgebung(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PFAD", tmp_path / "musik.db")
    monkeypatch.setattr(config, "MEDIA_ROOT", tmp_path / "media")
    monkeypatch.setattr(lernen, "trainieren_im_hintergrund", lambda: None)
    (tmp_path / "media").mkdir()
    con = db.verbinden()
    pfad = _datei(tmp_path / "media" / "lied", ".mp3")
    st = os.stat(pfad)
    lid = con.execute("INSERT INTO lieder(pfad, status, groesse, mtime, dauer, codec) VALUES ('lied.mp3','analysiert',?,?,3,'mp3')",
                      (st.st_size, st.st_mtime)).lastrowid
    con.commit()
    return con, pfad, lid


def test_api_stern_und_alle_schreiben_fuehrt_groesse_nach(umgebung):
    con, pfad, lid = umgebung
    with TestClient(web.app) as client:
        assert client.post("/api/stern", json={"lied_id": lid, "an": True}).json()["stern"] is True
        assert client.post("/api/stern", json={"lied_id": 999999, "an": True}).status_code == 404
    assert con.execute("SELECT stern FROM lieder WHERE id=?", (lid,)).fetchone()[0] == 1
    assert len(tags.ausstehend(con)) == 1
    assert tags.alle_schreiben(con, ausgabe=lambda *_: None) == 1
    st = os.stat(pfad)
    r = con.execute("SELECT groesse, mtime, stern_in_datei FROM lieder WHERE id=?", (lid,)).fetchone()
    assert (r[0], r[1], r[2]) == (st.st_size, st.st_mtime, 1)   # nächtliche Suche sieht keine Änderung
    assert tags.lesen(pfad) and tags.ausstehend(con) == []


def test_schreibgeschuetzt_bricht_sauber_ab(umgebung):
    con, pfad, lid = umgebung
    con.execute("UPDATE lieder SET stern=1 WHERE id=?", (lid,))
    con.commit()
    os.chmod(pfad, 0o444)
    meldungen = []
    try:
        assert tags.alle_schreiben(con, ausgabe=meldungen.append) == 0
    finally:
        os.chmod(pfad, 0o644)
    assert "nur lesend" in meldungen[-1] and len(tags.ausstehend(con)) == 1


def test_stern_wandert_bei_duplikaten_zur_behaltenen_datei(umgebung, tmp_path):
    con, pfad, lid = umgebung
    (tmp_path / "media" / "kopie.mp3").write_bytes(pfad.read_bytes())
    kopie = con.execute("INSERT INTO lieder(pfad, status, stern) VALUES ('kopie.mp3','analysiert',1)").lastrowid
    g = con.execute("INSERT INTO dup_gruppen(erstellt, status) VALUES (0,'freigegeben')").lastrowid
    con.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,?)", [(g, lid, 1), (g, kopie, 0)])
    papierkorb.gruppe_ausfuehren(con, g)
    con.commit()
    assert con.execute("SELECT stern FROM lieder WHERE id=?", (lid,)).fetchone()[0] == 1


@pytest.mark.parametrize("endung", [".mp3", ".flac", ".m4a", ".ogg"])
def test_sprache_in_datei_ohne_stern_zu_aendern(tmp_path, endung):
    pfad = _datei(tmp_path / "lied", endung)
    tags.schreiben(pfad, True)
    tags.schreiben(pfad, sprache="deu")
    t = mutagen.File(pfad).tags
    wert = {".mp3": lambda: str(t["TLAN"]), ".flac": lambda: t["LANGUAGE"][0], ".ogg": lambda: t["LANGUAGE"][0],
            ".m4a": lambda: bytes(t["----:com.apple.iTunes:LANGUAGE"][0]).decode()}[endung]()
    assert wert == "deu"
    assert tags.lesen(pfad)   # Stern blieb


def test_sprache_nur_bei_sicherer_erkennung_ausstehend(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_ROOT", tmp_path)
    con = db.verbinden(tmp_path / "t.db")
    _datei(tmp_path / "a", ".mp3"), _datei(tmp_path / "b", ".mp3")
    con.execute("INSERT INTO lieder(pfad, status, sprache, sprache_p) VALUES ('a.mp3', 'analysiert', 'de', 0.95)")
    con.execute("INSERT INTO lieder(pfad, status, sprache, sprache_p) VALUES ('b.mp3', 'analysiert', 'en', 0.4)")
    con.commit()
    assert [(x[1], x[3]) for x in tags.ausstehend(con)] == [("a.mp3", "deu")]
    assert tags.alle_schreiben(con, ausgabe=lambda *_: None) == 1
    assert tags.ausstehend(con) == []
    assert str(mutagen.File(tmp_path / "a.mp3").tags["TLAN"]) == "deu"


@pytest.mark.parametrize("endung", [".mp3", ".flac", ".m4a", ".ogg"])
def test_bpm_ueberschreibt_vorhandenen_tag(tmp_path, endung):
    pfad = _datei(tmp_path / "lied", endung)
    tags.schreiben(pfad, True, bpm=70)
    tags.schreiben(pfad, bpm=140)
    t = mutagen.File(pfad).tags
    wert = {".mp3": lambda: str(t["TBPM"]), ".flac": lambda: t["BPM"][0], ".ogg": lambda: t["BPM"][0],
            ".m4a": lambda: str(t["tmpo"][0])}[endung]()
    assert wert == "140"
    assert tags.lesen(pfad)   # Stern blieb


def test_bpm_nur_vom_aktuellen_tempo_modell(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEDIA_ROOT", tmp_path)
    con = db.verbinden(tmp_path / "t.db")
    _datei(tmp_path / "a", ".mp3"), _datei(tmp_path / "b", ".mp3")
    con.execute("INSERT INTO lieder(pfad, status, bpm, tempo_stand) VALUES ('a.mp3', 'analysiert', 127.6, ?)",
                (config.TEMPO_STAND,))
    con.execute("INSERT INTO lieder(pfad, status, bpm) VALUES ('b.mp3', 'analysiert', 93.0)")  # alter Schätzer
    con.commit()
    assert [(x[1], x[4]) for x in tags.ausstehend(con)] == [("a.mp3", 128)]
    assert tags.alle_schreiben(con, ausgabe=lambda *_: None) == 1
    assert tags.ausstehend(con) == []
    assert str(mutagen.File(tmp_path / "a.mp3").tags["TBPM"]) == "128"
