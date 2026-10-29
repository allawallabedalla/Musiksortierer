"""Papierkorb: verschieben, ausführen, zurückholen, erst nach Frist Datei für Datei löschen."""
import datetime

import pytest

from musiksort import config, db, papierkorb


@pytest.fixture
def umgebung(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PFAD", tmp_path / "musik.db")
    monkeypatch.setattr(config, "MEDIA_ROOT", tmp_path / "media")
    (tmp_path / "media" / "Musik").mkdir(parents=True)
    con = db.verbinden()
    ids = []
    for name in ("a.mp3", "b.mp3"):
        (tmp_path / "media" / "Musik" / name).write_bytes(b"x" * 10)
        ids.append(con.execute("INSERT INTO lieder(pfad, status) VALUES (?, 'analysiert')", (f"Musik/{name}",)).lastrowid)
    g = con.execute("INSERT INTO dup_gruppen(erstellt, status) VALUES (0, 'freigegeben')").lastrowid
    con.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,?)",
                    [(g, ids[0], 1), (g, ids[1], 0)])
    con.commit()
    return con, tmp_path / "media", ids, g


def test_ausfuehren_verschiebt_nur_vorgemerkte_und_suche_laesst_sie_in_ruhe(umgebung):
    con, media, (a, b), g = umgebung
    assert papierkorb.gruppe_ausfuehren(con, g) == 1
    con.commit()
    assert (media / "Musik/a.mp3").exists() and not (media / "Musik/b.mp3").exists()
    neu = con.execute("SELECT pfad, status FROM lieder WHERE id=?", (b,)).fetchone()
    assert neu[1] == "papierkorb" and neu[0].startswith("_papierkorb/") and (media / neu[0]).exists()
    assert con.execute("SELECT status FROM dup_gruppen WHERE id=?", (g,)).fetchone()[0] == "erledigt"
    assert config.ausgeschlossen(neu[0])
    from musiksort.scan import gruppen_ungueltig_machen
    gruppen_ungueltig_machen(con)
    assert con.execute("SELECT status FROM dup_gruppen WHERE id=?", (g,)).fetchone()[0] == "erledigt"


def test_zurueckholen(umgebung):
    con, media, (a, b), g = umgebung
    papierkorb.gruppe_ausfuehren(con, g)
    papierkorb.zurueckholen(con, b)
    con.commit()
    assert (media / "Musik/b.mp3").exists()
    assert tuple(con.execute("SELECT pfad, status FROM lieder WHERE id=?", (b,)).fetchone()) == ("Musik/b.mp3", "analysiert")
    assert con.execute("SELECT status FROM dup_gruppen WHERE id=?", (g,)).fetchone()[0] == "ersetzt"


def test_leeren_erst_nach_frist(umgebung):
    con, media, (a, b), g = umgebung
    vor40 = datetime.date.today() - datetime.timedelta(days=40)
    papierkorb.verschieben(con, b, "Test", heute=vor40)
    con.commit()
    assert papierkorb.leeren(con, tage=60) == 0          # Frist noch nicht um
    assert papierkorb.leeren(con, tage=30) == 1
    con.commit()
    assert con.execute("SELECT status FROM lieder WHERE id=?", (b,)).fetchone()[0] == "geloescht"
    assert not (media / "_papierkorb" / vor40.isoformat()).exists()   # leere Ordner aufgeräumt
    assert (media / "Musik/a.mp3").exists()                          # Behaltenes unberührt


def test_bewertung_wandert_zur_behaltenen_datei(umgebung):
    con, media, (a, b), g = umgebung
    rock = con.execute("INSERT INTO genres(name) VALUES ('Rock')").lastrowid
    zusatz = con.execute("INSERT INTO genres(name) VALUES ('Live')").lastrowid
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',5)", (b, rock))   # nur die Kopie war bewertet
    con.execute("INSERT INTO etiketten VALUES (?,?,0,'ich',5)", (b, zusatz))
    papierkorb.gruppe_ausfuehren(con, g)
    assert sorted(tuple(r) for r in con.execute("SELECT genre_id, haupt FROM etiketten WHERE lied_id=?", (a,))) \
        == sorted([(rock, 1), (zusatz, 0)])


def test_eigene_bewertung_der_behaltenen_bleibt(umgebung):
    con, media, (a, b), g = umgebung
    pop, rock = (con.execute("INSERT INTO genres(name) VALUES (?)", (n,)).lastrowid for n in ("Pop", "Rock"))
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',5)", (a, pop))
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',9)", (b, rock))
    papierkorb.gruppe_ausfuehren(con, g)
    assert [r[0] for r in con.execute("SELECT genre_id FROM etiketten WHERE lied_id=?", (a,))] == [pop]
