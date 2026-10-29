"""Automatische Duplikat-Entscheidung: jede Stufe, Trockenlauf, Zurücknehmen, Konflikte."""
import shutil
import subprocess

import numpy as np
import pytest

from musiksort import config, db, dupauto, duplikate


@pytest.fixture
def umgebung(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PFAD", tmp_path / "musik.db")
    monkeypatch.setattr(config, "MEDIA_ROOT", tmp_path / "media")
    monkeypatch.setattr(config, "MOUNT_PRUEFEN", False)
    monkeypatch.setattr(config, "env", lambda name, standard=None: None)  # kein AcoustID-Schlüssel
    (tmp_path / "media").mkdir()
    return db.verbinden(), tmp_path / "media"


def _audio(ziel, teile, bitrate="128k"):
    """Datei aus Abschnitten [(sekunden, frequenz)], damit Anfang/Ende gezielt gleich oder verschieden sind."""
    ziel.parent.mkdir(parents=True, exist_ok=True)
    quellen = []
    for i, (sek, f) in enumerate(teile):
        quellen += ["-f", "lavfi", "-i", f"aevalsrc='0.5*sin(2*PI*{f}*t)*(0.6+0.4*sin(2*PI*0.7*t))':d={sek}"]
    filt = "".join(f"[{i}]" for i in range(len(teile))) + f"concat=n={len(teile)}:v=0:a=1"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *quellen, "-filter_complex", filt, "-b:a", bitrate, str(ziel)],
                   check=True)


def _lied(con, pfad, media, **werte):
    datei = media / pfad
    groesse = datei.stat().st_size if datei.exists() else werte.pop("groesse", 1000)
    standard = {"codec": "mp3", "bitrate": 128, "verlustfrei": 0, "dauer": 200.0, "samplerate": 44100,
                "fp": np.arange(300, dtype=np.uint32).tobytes(), "mtime": 1.0}
    standard.update(werte)
    spalten = ["pfad", "status", "groesse"] + list(standard)
    return con.execute(f"INSERT INTO lieder({','.join(spalten)}) VALUES ({','.join('?' * len(spalten))})",
                       [pfad, "analysiert", groesse, *standard.values()]).lastrowid


def _gruppe(con, ids, sicher=1, abweichung=0.0):
    g = con.execute("INSERT INTO dup_gruppen(erstellt, sicher) VALUES (0, ?)", (sicher,)).lastrowid
    con.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten, abweichung) VALUES (?,?,?,?)",
                    [(g, i, int(k == 0), 0.0 if k == 0 else abweichung) for k, i in enumerate(ids)])
    con.commit()
    return g


def _stand(con, g):
    r = con.execute("SELECT status, entschieden_von FROM dup_gruppen WHERE id=?", (g,)).fetchone()
    beh = con.execute("SELECT l.pfad FROM dup_mitglieder m JOIN lieder l ON l.id=m.lied_id "
                      "WHERE m.gruppe_id=? AND m.behalten=1", (g,)).fetchone()[0]
    return r[0], r[1], beh


def test_stufe_1_bitgleich_mit_ordnerregel(umgebung, monkeypatch):
    con, media = umgebung
    monkeypatch.setattr(config, "ORDNER_RANG", {"Sortiert": 5, "Doppelt": 0})
    _audio(media / "Doppelt/a.mp3", [(20, 440)])
    (media / "Sortiert").mkdir()
    shutil.copy(media / "Doppelt/a.mp3", media / "Sortiert/a.mp3")
    a = _lied(con, "Doppelt/a.mp3", media)
    b = _lied(con, "Sortiert/a.mp3", media)
    g = _gruppe(con, [a, b], abweichung=0.1)   # nicht „eng“: nur die Prüfsumme kann entscheiden
    dupauto.laufen(con, stufen=["1"], ausgabe=lambda *_: None)
    assert _stand(con, g) == ("freigegeben", "auto:1", "Sortiert/a.mp3")  # gepflegter Ordner gewinnt


def test_stufe_1_gleiche_groesse_anderer_inhalt_bleibt(umgebung):
    con, media = umgebung
    _audio(media / "x/a.mp3", [(20, 440)])
    _audio(media / "x/b.mp3", [(20, 550)])
    assert (media / "x/a.mp3").stat().st_size == (media / "x/b.mp3").stat().st_size  # CBR: gleiche Größe
    g = _gruppe(con, [_lied(con, "x/a.mp3", media), _lied(con, "x/b.mp3", media)], abweichung=0.1)
    dupauto.laufen(con, stufen=["1"], ausgabe=lambda *_: None)
    assert _stand(con, g)[0] == "offen"


def test_stufe_2_eng_beste_qualitaet(umgebung):
    con, media = umgebung
    for name, rate in (("x/a.mp3", "128k"), ("x/b.mp3", "320k")):
        _audio(media / name, [(150, 440)], rate)
    _audio(media / "y/a.mp3", [(150, 440)])
    _audio(media / "y/b.mp3", [(150, 440), (2, 880)])              # 2 s mehr Ton: nicht eng
    _audio(media / "s/a.mp3", [(150, 440)])
    _audio(media / "s/b.mp3", [(150, 440), (4, 0)])                # 4 s Stille am Ende: eng ohne Stille
    _audio(media / "e/a.mp3", [(170, 440), (130, 660)])
    _audio(media / "e/b.mp3", [(170, 440), (130, 990)])            # gleicher Anfang, anderes Ende
    g = _gruppe(con, [_lied(con, "x/a.mp3", media, bitrate=128, dauer=150.0),
                      _lied(con, "x/b.mp3", media, bitrate=320, dauer=150.6)], abweichung=0.03)
    locker = _gruppe(con, [_lied(con, "y/a.mp3", media, dauer=150), _lied(con, "y/b.mp3", media, dauer=152)],
                     abweichung=0.03)
    still = _gruppe(con, [_lied(con, "s/a.mp3", media, dauer=150), _lied(con, "s/b.mp3", media, dauer=154)],
                    abweichung=0.03)
    ende = _gruppe(con, [_lied(con, "e/a.mp3", media, dauer=300), _lied(con, "e/b.mp3", media, dauer=300.2)],
                   abweichung=0.01)
    dupauto.laufen(con, stufen=["2"], ausgabe=lambda *_: None)
    assert _stand(con, g) == ("freigegeben", "auto:2", "x/b.mp3")
    assert _stand(con, locker)[0] == "offen"
    assert _stand(con, still)[:2] == ("freigegeben", "auto:2")
    assert _stand(con, ende)[0] == "offen"   # Anfang gleich reicht nicht mehr


def test_stufe_c_namen_und_laenge(umgebung):
    con, media = umgebung
    g1 = _gruppe(con, [_lied(con, "x/Song.mp3", media, groesse=1),
                       _lied(con, "x/Song (Radio Edit).mp3", media, dauer=201.5, groesse=2)], abweichung=0.1)
    g2 = _gruppe(con, [_lied(con, "y/Lied.mp3", media, groesse=3), _lied(con, "y/Lied2.mp3", media, dauer=215, groesse=4)],
                 sicher=0, abweichung=0.2)
    g3 = _gruppe(con, [_lied(con, "z/Gleich.mp3", media, groesse=5), _lied(con, "z/Gleich (1).mp3", media, dauer=202, groesse=6)],
                 abweichung=0.1)
    dupauto.laufen(con, stufen=["C"], ausgabe=lambda *_: None)
    assert _stand(con, g1)[:2] == ("verworfen", "auto:C")
    assert _stand(con, g2)[:2] == ("verworfen", "auto:C")
    assert _stand(con, g3)[0] == "offen"   # „(1)“ ist keine andere Fassung


def test_stufe_a_ende_entscheidet(umgebung):
    con, media = umgebung
    _audio(media / "a/gut.mp3", [(170, 440), (130, 660)], "320k")
    _audio(media / "a/schlecht.mp3", [(170, 440), (130, 660)], "96k")
    _audio(media / "b/eins.mp3", [(170, 440), (130, 660)])
    _audio(media / "b/anderes_ende.mp3", [(170, 440), (130, 990)])
    gleich = _gruppe(con, [_lied(con, "a/gut.mp3", media, bitrate=320, dauer=300),
                           _lied(con, "a/schlecht.mp3", media, bitrate=96, dauer=301.5)], abweichung=0.08)
    anders = _gruppe(con, [_lied(con, "b/eins.mp3", media, dauer=300),
                           _lied(con, "b/anderes_ende.mp3", media, dauer=301.5)], abweichung=0.08)
    zaehler = dupauto.laufen(con, stufen=["A"], ausgabe=lambda *_: None)
    assert _stand(con, gleich) == ("freigegeben", "auto:A", "a/gut.mp3")
    assert _stand(con, anders)[0] == "offen"
    assert not any(k[1] in ("Lesefehler", "Fehler") for k in zaehler)   # offen wegen des Vergleichs, nicht wegen eines Fehlers
    enden = [np.frombuffer(r[0], np.uint32) for r in con.execute(
        "SELECT fp_ende2 FROM lieder WHERE pfad LIKE 'b/%' ORDER BY pfad")]
    assert duplikate.bitfehler(*enden) > duplikate.MAX_BITFEHLER


def test_stufe_b_acoustid_nachgestellt(umgebung, monkeypatch):
    con, media = umgebung
    monkeypatch.setattr(config, "env", lambda name, standard=None: "schluessel" if name == "ACOUSTID_KEY" else standard)
    rec = {}

    def nachschlagen(k, fp, dauer, meta):
        assert k == "schluessel" and meta == "recordings" and isinstance(fp, (str, bytes))
        return {"status": "ok", "results": [{"score": 0.95, "recordings": [{"id": rec[dauer], "title": "Lied"}]}]}
    a = _lied(con, "x/a.mp3", media, dauer=200, groesse=1)
    b = _lied(con, "x/b.mp3", media, dauer=203, groesse=2)
    c = _lied(con, "y/c.mp3", media, dauer=210, groesse=3)
    d = _lied(con, "y/d.mp3", media, dauer=225, groesse=4)
    rec.update({200: "R1", 203: "R1", 210: "R2", 225: "R3"})
    gleich = _gruppe(con, [a, b], abweichung=0.1)
    verschieden = _gruppe(con, [c, d], sicher=0, abweichung=0.2)
    monkeypatch.setattr(dupauto.time, "sleep", lambda s: None)
    dupauto.laufen(con, stufen=["B"], nachschlagen=nachschlagen, ausgabe=lambda *_: None)
    assert _stand(con, gleich)[:2] == ("freigegeben", "auto:B")
    assert _stand(con, verschieden)[:2] == ("verworfen", "auto:B")


def test_stufe_d_lernt_aus_deinen_entscheidungen(umgebung):
    con, media = umgebung
    rng = np.random.default_rng(3)
    n = 0
    for i in range(240):   # deine Regel: Länge ≥ 5 s anders → alle behalten, sonst beste behalten
        dd = float(rng.uniform(0, 12))
        ids = [_lied(con, f"h/{i}a.mp3", media, dauer=200, groesse=n + 1, bitrate=320),
               _lied(con, f"h/{i}b.mp3", media, dauer=200 + dd, groesse=n + 2, bitrate=128)]
        n += 2
        g = _gruppe(con, ids, abweichung=0.1)
        con.execute("BEGIN IMMEDIATE")
        duplikate.entscheiden(con, g, "verworfen" if dd >= 5 else "freigegeben", ids, ids[0], von="ich")
        con.commit()
    kurz = _gruppe(con, [_lied(con, "o/a.mp3", media, groesse=9001, bitrate=320),
                         _lied(con, "o/b.mp3", media, dauer=200.3, groesse=9002)], abweichung=0.1)
    lang = _gruppe(con, [_lied(con, "o/c.mp3", media, groesse=9003, bitrate=320),
                         _lied(con, "o/d.mp3", media, dauer=211, groesse=9004)], abweichung=0.1)
    dupauto.laufen(con, stufen=["D"], ausgabe=lambda *_: None)
    assert _stand(con, kurz)[:2] == ("freigegeben", "auto:D")
    assert _stand(con, lang)[:2] == ("verworfen", "auto:D")


def test_stufe_d_ohne_genug_entscheidungen_still(umgebung):
    con, media = umgebung
    g = _gruppe(con, [_lied(con, "o/a.mp3", media, groesse=1), _lied(con, "o/b.mp3", media, groesse=2)], abweichung=0.1)
    dupauto.laufen(con, stufen=["D"], ausgabe=lambda *_: None)
    assert _stand(con, g)[0] == "offen"


def test_trockenlauf_aendert_nichts_und_zuruecknehmen(umgebung):
    con, media = umgebung
    _audio(media / "x/a.mp3", [(40, 440)])
    _audio(media / "x/b.mp3", [(40, 440)], "320k")
    g = _gruppe(con, [_lied(con, "x/a.mp3", media, dauer=40), _lied(con, "x/b.mp3", media, bitrate=320, dauer=40)],
                abweichung=0.02)
    z = dupauto.laufen(con, trocken=True, ausgabe=lambda *_: None)
    assert z[("2", "freigegeben")] == 1 and _stand(con, g)[0] == "offen"
    dupauto.laufen(con, stufen=["2"], ausgabe=lambda *_: None)
    assert dupauto.uebersicht(con) == {"2": {"freigegeben": 1}}
    con.execute("BEGIN IMMEDIATE")
    assert dupauto.alle_zuruecknehmen(con) == 1
    con.commit()
    assert _stand(con, g)[:2] == ("offen", None) and dupauto.uebersicht(con) == {}


def test_automatik_haelt_die_sicherheitsregeln_ein(umgebung):
    """X ist in einer offenen Gruppe die Empfehlung; eine zweite Gruppe darf X nicht automatisch wegnehmen."""
    con, media = umgebung
    for name, rate in (("x/x.mp3", "192k"), ("x/y.mp3", "128k"), ("x/z.mp3", "320k")):
        _audio(media / name, [(40, 440)], rate)
    x = _lied(con, "x/x.mp3", media, bitrate=192, dauer=40)
    y = _lied(con, "x/y.mp3", media, bitrate=128, dauer=40)
    z = _lied(con, "x/z.mp3", media, bitrate=320, dauer=40.2)
    _gruppe(con, [x, y], abweichung=0.3)                  # offen, X behalten (keine Stufe zuständig)
    g2 = _gruppe(con, [x, z], abweichung=0.02)            # eng: Stufe 2 will Z behalten und X vormerken
    zaehler = dupauto.laufen(con, stufen=["2"], ausgabe=lambda *_: None)
    assert zaehler[("2", "Konflikt")] == 1 and _stand(con, g2)[0] == "offen"


def test_stufe_b_titel_nennen_verschiedene_fassungen(umgebung, monkeypatch):
    con, media = umgebung
    monkeypatch.setattr(config, "env", lambda name, standard=None: "schluessel" if name == "ACOUSTID_KEY" else standard)
    antworten = {200: [("R1", "Lied (Radio Edit)"), ("R9", "Lied")], 204: [("R2", "Lied (Extended Mix)"), ("R9", "Lied")]}

    def nachschlagen(k, fp, dauer, meta):
        return {"status": "ok", "results": [{"score": 0.9 if i else 0.95, "recordings": [{"id": r, "title": t}]}
                                            for i, (r, t) in enumerate(antworten[dauer])]}
    monkeypatch.setattr(dupauto.time, "sleep", lambda s: None)
    g = _gruppe(con, [_lied(con, "x/a.mp3", media, dauer=200, groesse=1), _lied(con, "x/b.mp3", media, dauer=204, groesse=2)],
                sicher=0, abweichung=0.2)   # teilen sich R9, wahrscheinlichste Aufnahmen sind aber Edit/Extended
    dupauto.laufen(con, stufen=["B"], nachschlagen=nachschlagen, ausgabe=lambda *_: None)
    assert _stand(con, g)[:2] == ("verworfen", "auto:B")


def test_falsche_flac_verliert_gegen_echte_320er(umgebung):
    con, media = umgebung
    _audio(media / "f/echt.mp3", [(30, 440)], "320k")
    rate = 44100   # Rauschen mit harter Kante bei 16 kHz, wie es ein MP3-Kodierer hinterlässt
    roh = np.random.default_rng(1).standard_normal(rate * 30) * 0.1
    spek = np.fft.rfft(roh)
    spek[np.fft.rfftfreq(roh.size, 1 / rate) > 16000] = 0
    pcm = np.fft.irfft(spek, roh.size).astype(np.float32).tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", "1", "-i", "-",
                    str(media / "f/aufgeblasen.flac")], input=pcm, check=True)
    echt = _lied(con, "f/echt.mp3", media, bitrate=320, dauer=30)
    flac = _lied(con, "f/aufgeblasen.flac", media, codec="flac", verlustfrei=1, bitrate=900, dauer=30)
    ms = [dict(r) for r in con.execute("SELECT * FROM lieder WHERE id IN (?,?)", (echt, flac))]
    assert dupauto.beste_gemessen(con, ms)["id"] == echt
    gf, kante = con.execute("SELECT grenzfrequenz, kante FROM lieder WHERE id=?", (flac,)).fetchone()
    assert 14000 < gf < 17500 and kante >= 25
