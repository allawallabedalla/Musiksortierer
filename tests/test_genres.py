"""Genre-Baum: Namensabgleich, Umbenennen, Verschieben, Zusammenführen, Verwechslungen."""
import numpy as np
import pytest

from musiksort import db, genres, lernen


@pytest.fixture
def con(tmp_path):
    return db.verbinden(tmp_path / "t.db")


def _genre(con, name, eltern=None):
    return con.execute("INSERT INTO genres(name, eltern_id) VALUES (?,?)", (name, eltern)).lastrowid


def _lied(con, pfad):
    return con.execute("INSERT INTO lieder(pfad) VALUES (?)", (pfad,)).lastrowid


def test_aehnliche_schreibweisen():
    assert genres.aehnlich("Hard Techno", "Hardtechno")
    assert genres.aehnlich("Schlager", "schlager")
    assert genres.aehnlich("Hörspiel", "Hoerspiel")
    assert not genres.aehnlich("Techno", "House")


def test_namen_vorschlag_warnt_und_kennt_discogs(con):
    _genre(con, "Hard Techno")
    t = genres.namen_vorschlagen(con, "hardtechno", {})
    assert t["warnung"] and "Hard Techno" in t["warnung"]
    t = genres.namen_vorschlagen(con, "dub tech", {})
    assert any(d["name"] == "Dub Techno" for d in t["discogs"])
    assert genres.namen_vorschlagen(con, "", {})["warnung"] is None


def test_verschieben_ohne_kreis(con):
    a = _genre(con, "Elektronisch")
    b = _genre(con, "Techno", a)
    c = _genre(con, "Hypnotic", b)
    with pytest.raises(genres.GenreFehler):
        genres.aendern(con, a, "Elektronisch", c)  # unter den eigenen Enkel
    genres.aendern(con, c, "Hypnotic Techno", a)
    assert tuple(con.execute("SELECT name, eltern_id FROM genres WHERE id=?", (c,)).fetchone()) == ("Hypnotic Techno", a)
    _genre(con, "House", a)
    with pytest.raises(genres.GenreFehler):
        genres.aendern(con, b, "House", a)  # Name dort schon vergeben


def test_zusammenfuehren_nimmt_bewertungen_und_kinder_mit(con):
    oben = _genre(con, "Elektronisch")
    deep = _genre(con, "Deep House", oben)
    lofi = _genre(con, "Lo-Fi House", oben)
    lofi_kind = _genre(con, "Chill", lofi)
    deep_kind = _genre(con, "Chill", deep)    # gleicher Name unter beiden: verschmilzt mit
    l1, l2, l3 = _lied(con, "1.mp3"), _lied(con, "2.mp3"), _lied(con, "3.mp3")
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (l1, lofi))
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (l2, deep))
    con.execute("INSERT INTO etiketten VALUES (?,?,0,'ich',0)", (l2, lofi))   # Tag auf beiden
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (l3, lofi_kind))
    con.commit()
    genres.zusammenfuehren(con, lofi, deep)
    assert con.execute("SELECT 1 FROM genres WHERE id IN (?,?)", (lofi, lofi_kind)).fetchone() is None
    et = {(r[0], r[1]): r[2] for r in con.execute("SELECT lied_id, genre_id, haupt FROM etiketten")}
    assert et == {(l1, deep): 1, (l2, deep): 1, (l3, deep_kind): 1}


def test_zusammenfuehren_nicht_in_eigenes_kind(con):
    a = _genre(con, "Techno")
    b = _genre(con, "Hypnotic", a)
    with pytest.raises(genres.GenreFehler):
        genres.zusammenfuehren(con, a, b)
    genres.zusammenfuehren(con, b, a)  # andersherum ist es in Ordnung
    assert con.execute("SELECT COUNT(*) FROM genres").fetchone()[0] == 1


def test_verwechslungen():
    y = np.array([1] * 10 + [2] * 10 + [3] * 10)
    vorh = y.copy()
    vorh[:4] = 2      # 4 von „1“ landen bei „2“
    vorh[10:13] = 1   # 3 von „2“ landen bei „1“
    vorh[20] = 1      # 1 von „3“ landet bei „1“: unter der Schwelle
    erg = lernen.verwechslungen_finden(y, vorh)
    assert erg == [(1, 2, 0.35, 20)]


def test_zusammenfuehren_ins_elternteil_mit_gleichnamigem_kind(con):
    techno = _genre(con, "Techno")
    hyp = _genre(con, "Hypnotic", techno)
    hyp2 = _genre(con, "Hypnotic", hyp)
    deep = _genre(con, "Deep", hyp2)
    l1, l2 = _lied(con, "1.mp3"), _lied(con, "2.mp3")
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (l1, hyp))
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (l2, hyp2))
    con.commit()
    genres.zusammenfuehren(con, hyp, techno)
    # keine Etiketten auf gelöschte Genres, kein verwaister Zweig
    verwaist = con.execute("SELECT COUNT(*) FROM etiketten WHERE genre_id NOT IN (SELECT id FROM genres)").fetchone()[0]
    assert verwaist == 0
    baum = {r[0]: r[1] for r in con.execute("SELECT id, eltern_id FROM genres")}
    assert baum[hyp2] == techno and baum[deep] == hyp2 and hyp not in baum


def test_genre_ids_nie_wiederverwendet_und_stichproben_wandern_mit(con):
    a = _genre(con, "House")
    b = _genre(con, "Trance")
    l = _lied(con, "1.mp3")
    con.execute("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)", (l, b, b))
    con.commit()
    genres.zusammenfuehren(con, b, a)
    neu = _genre(con, "Ambient")
    assert neu > b                                      # keine wiederverwendete ID
    assert con.execute("SELECT vorhergesagt, gewaehlt FROM stichproben").fetchone()[:] == (a, a)
    assert con.execute("SELECT wert FROM modell_info WHERE schluessel='genre_stand'").fetchone()[0] == "1"


def test_alte_db_bekommt_autoincrement(tmp_path):
    import sqlite3
    alt = sqlite3.connect(tmp_path / "alt.db")
    alt.execute("CREATE TABLE genres (id INTEGER PRIMARY KEY, name TEXT NOT NULL, eltern_id INTEGER, erstellt REAL, UNIQUE(name, eltern_id))")
    alt.execute("INSERT INTO genres(id, name) VALUES (5, 'Rock')")
    alt.commit()
    alt.close()
    con = db.verbinden(tmp_path / "alt.db")
    assert "AUTOINCREMENT" in con.execute("SELECT sql FROM sqlite_master WHERE name='genres'").fetchone()[0]
    assert con.execute("SELECT name FROM genres WHERE id=5").fetchone()[0] == "Rock"
    con.execute("DELETE FROM genres WHERE id=5")
    assert con.execute("INSERT INTO genres(name) VALUES ('Pop')").lastrowid == 6
