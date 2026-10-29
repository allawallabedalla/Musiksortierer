"""Startmodell: anonym exportieren, in eine neue Installation laden, dort nur fürs Modell nutzen."""
import json
import zipfile

import numpy as np
import pytest

from musiksort import db, genres as genrelib, lernen, startmodell


@pytest.fixture
def zwei(tmp_path, monkeypatch):
    monkeypatch.setattr(lernen, "bestand", lernen.Bestand())
    return db.verbinden(tmp_path / "alt.db"), db.verbinden(tmp_path / "neu.db"), tmp_path / "startmodell"


def _lieder(con, zentren, n, rng, praefix):
    ids = []
    for w, z in enumerate(zentren):
        for i in range(n):
            v = z + 0.6 * rng.normal(size=1280)
            v = (v / np.linalg.norm(v)).astype(np.float16)
            ids.append((con.execute("INSERT INTO lieder(pfad, status, emb, dauer, interpret, titel) "
                                    "VALUES (?, 'analysiert', ?, 200.4, 'Geheimer Interpret', 'Geheimer Titel')",
                                    (f"Privat/{praefix}{w}_{i}.mp3", v.tobytes())).lastrowid, w))
    con.commit()
    return ids


def test_export_ohne_personenbezug_und_laden(zwei):
    alt, neu, ordner = zwei
    rng = np.random.default_rng(3)
    zentren = rng.normal(size=(3, 1280))
    oben = alt.execute("INSERT INTO genres(name) VALUES ('Electronic')").lastrowid
    gids = [alt.execute("INSERT INTO genres(name, eltern_id) VALUES (?, ?)", (n, oben)).lastrowid
            for n in ("House", "Techno")] + [alt.execute("INSERT INTO genres(name) VALUES ('Hörbuch')").lastrowid]
    klein = alt.execute("INSERT INTO genres(name, eltern_id) VALUES ('Acid', ?)", (gids[1],)).lastrowid
    lieder = _lieder(alt, zentren, 20, rng, "a")
    for lid, w in lieder:
        alt.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (lid, gids[w]))
    extra = _lieder(alt, zentren[1:2], 2, rng, "x")   # zu klein: geht in „Techno“ auf
    for lid, _ in extra:
        alt.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (lid, klein))
    # Kopie eines bewerteten Lieds: zählt nur einmal
    kopie = alt.execute("INSERT INTO lieder(pfad, status, emb) SELECT 'Privat/kopie.mp3', 'analysiert', emb "
                        "FROM lieder WHERE id=?", (lieder[0][0],)).lastrowid
    alt.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (kopie, gids[0]))
    g = alt.execute("INSERT INTO dup_gruppen(erstellt) VALUES (0)").lastrowid
    alt.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,?)",
                    [(g, lieder[0][0], 1), (g, kopie, 0)])
    alt.commit()

    info = startmodell.exportieren(alt, ordner)
    assert info["beispiele"] == 62 and info["je_genre"]["Electronic › Techno"] == 22
    assert sorted(info["genres"]) == ["Electronic › House", "Electronic › Techno", "Hörbuch"]
    # keine Pfade, Interpreten, Titel oder Lied-IDs in irgendeiner Datei
    for datei in ordner.iterdir():
        roh = datei.read_bytes()
        if datei.suffix == ".npz":
            with zipfile.ZipFile(datei) as z:
                assert all(n.endswith(".npy") for n in z.namelist())
                roh = b"".join(z.read(n) for n in z.namelist())
            np.load(datei, allow_pickle=False)   # nur Zahlen, nichts Ausführbares
        for verboten in (b"Privat", b"Geheim", b".mp3"):
            assert verboten not in roh, (datei.name, verboten)
    assert set(json.loads((ordner / "info.json").read_text())) >= {"genres", "beispiele", "stand"}

    # neue Installation: Lieder ohne jede Bewertung, nur das Startmodell
    neue = _lieder(neu, zentren, 15, np.random.default_rng(9), "n")
    assert startmodell.laden(neu, ordner) == 62
    namen = {r[0] for r in neu.execute("SELECT name FROM genres")}
    assert {"Electronic", "House", "Techno", "Hörbuch"} <= namen
    lernen.trainieren(neu)
    pfade = {g: p for g, p in genrelib_pfade(neu).items()}
    erwartet = ["Electronic › House", "Electronic › Techno", "Hörbuch"]
    vorh = dict(neu.execute("SELECT lied_id, genre_id FROM vorhersagen").fetchall())
    treffer = sum(pfade[vorh[lid]] == erwartet[w] for lid, w in neue)
    assert treffer / len(neue) > 0.9
    # Freigaben nur aus eigenen Bewertungen: nichts wird selbst sortiert
    assert neu.execute("SELECT COUNT(*) FROM vorhersagen WHERE auto=1 OR ober_id IS NOT NULL").fetchone()[0] == 0
    # Zusammenführen nimmt die Startbeispiele mit
    house = neu.execute("SELECT id FROM genres WHERE name='House'").fetchone()[0]
    techno = neu.execute("SELECT id FROM genres WHERE name='Techno'").fetchone()[0]
    genrelib.zusammenfuehren(neu, house, techno)
    assert neu.execute("SELECT COUNT(*) FROM startbeispiele WHERE genre_id=?", (techno,)).fetchone()[0] == 42


def genrelib_pfade(con):
    return startmodell._pfade(con)[0]
