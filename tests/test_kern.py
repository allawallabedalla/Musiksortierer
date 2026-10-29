import numpy as np

from musiksort import config, duplikate, scan


def _ton(sekunden, laut):
    t = np.arange(int(sekunden * config.SR)) / config.SR
    return (laut * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_snippet_start_findet_vollen_teil():
    a = np.concatenate([_ton(60, 0.05), _ton(120, 0.8)])
    start, quelle = scan.snippet_start(a, 180)
    assert quelle == "energie"
    assert 45 <= start <= 62


def test_snippet_kurzes_lied_anteil():
    start, quelle = scan.snippet_start(_ton(35, 0.5), 35)
    assert (start, quelle) == (0.0, "anteil")


def test_snippet_rueckfall_ohne_energiesprung():
    a = np.zeros(int(200 * config.SR), np.float32)
    start, quelle = scan.snippet_start(a, 200)
    assert (start, quelle) == (105.0, "1:45")


def test_bitfehler_gleich_und_versetzt():
    rng = np.random.default_rng(0)
    fp = rng.integers(0, 2**32, 300, dtype=np.uint64).astype(np.uint32)
    assert duplikate.bitfehler(fp, fp) == 0
    assert duplikate.bitfehler(fp[5:], fp) == 0
    fremd = rng.integers(0, 2**32, 300, dtype=np.uint64).astype(np.uint32)
    assert duplikate.bitfehler(fp, fremd) > 0.4


def test_qualitaet_verlustfrei_vor_bitrate():
    flac = {"verlustfrei": 1, "bitrate": 900, "bittiefe": 16, "samplerate": 44100,
            "interpret": None, "titel": None, "album": None, "pfad": "a.flac"}
    mp3 = {"verlustfrei": 0, "bitrate": 320, "bittiefe": None, "samplerate": 44100,
           "interpret": "x", "titel": "y", "album": "z", "pfad": "a.mp3"}
    assert duplikate.qualitaet(flac) > duplikate.qualitaet(mp3)


def test_hauptteil_proben_begrenzt_und_laut():
    leise, laut = _ton(120, 0.02), _ton(600, 0.8)
    a = np.concatenate([leise, laut])
    proben, bpm_teil = scan.hauptteil_proben(a)
    assert proben.size == 5 * 30 * config.SR          # höchstens 150 s statt 12 min
    assert np.abs(proben).max() > 0.5 and np.sqrt((proben ** 2).mean()) > 0.4   # nur aus dem lauten Teil
    assert bpm_teil.size == 60 * config.SR
    kurz = _ton(100, 0.5)
    assert scan.hauptteil_proben(kurz)[0].size == kurz.size


def test_ausschluss_auch_tiefer(monkeypatch):
    monkeypatch.setattr(config, "AUSGESCHLOSSEN", config.AUSGESCHLOSSEN | {"Videos"})
    monkeypatch.setattr(config, "AUSGESCHLOSSENE_PFADE", {"Archiv/x/geraeusche"})
    assert config.ausgeschlossen("Videos/x.mp3") and config.ausgeschlossen("@eaDir/x.mp3")
    assert config.ausgeschlossen("Archiv/x/geraeusche/sound/a.wav")
    assert not config.ausgeschlossen("Archiv/x/lieder/a.mp3")
    assert not config.ausgeschlossen("Archiv/x/geraeuscheX/a.mp3")


def test_sicherung_behaelt_sieben(tmp_path, monkeypatch):
    from musiksort import db, sicherung
    monkeypatch.setattr(config, "DB_PFAD", tmp_path / "musik.db")
    c = db.verbinden()
    c.execute("INSERT INTO genres(name) VALUES ('Techno')")
    c.commit()
    ordner = tmp_path / "sicherung"
    ordner.mkdir()
    for i in range(1, 10):
        (ordner / f"musik-2026-01-0{i}.db").write_bytes(b"alt")
    ziel = sicherung.sichern(ordner)
    import sqlite3
    assert sqlite3.connect(ziel).execute("SELECT name FROM genres").fetchone()[0] == "Techno"
    assert len(list(ordner.glob("musik-*.db"))) == 7


def test_bewertungen_export_enthaelt_eigene_arbeit(tmp_path, monkeypatch):
    import json
    from musiksort import config, db, sicherung
    monkeypatch.setattr(config, "DB_PFAD", tmp_path / "musik.db")
    con = db.verbinden()
    lid = con.execute("INSERT INTO lieder(pfad, status) VALUES ('a.mp3', 'analysiert')").lastrowid
    gid = con.execute("INSERT INTO genres(name) VALUES ('Pop')").lastrowid
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (lid, gid))
    con.commit()
    pfad, n = sicherung.bewertungen_exportieren(tmp_path / "b.json")
    d = json.loads(pfad.read_text())
    assert n == 1 and d["genres"][0]["name"] == "Pop" and [(x["id"], x["pfad"], x["stern"]) for x in d["lieder"]] == [(lid, "a.mp3", 0)]


def test_meldung_ohne_zugang_nur_journal(monkeypatch, capsys):
    from musiksort import meldung
    monkeypatch.setattr(config, "env", lambda name, standard=None: None)
    assert meldung.melden("x.service") is False and "nur ins Journal" in capsys.readouterr().out
