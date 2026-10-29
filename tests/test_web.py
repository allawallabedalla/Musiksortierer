"""Web-API: Bewerten nur mit gültigen Genres, Snippets parallel, Duplikat-Entscheidung."""
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from musiksort import config, db, lernen, web


@pytest.fixture
def umgebung(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PFAD", tmp_path / "musik.db")
    monkeypatch.setattr(config, "SNIPPETS", tmp_path / "snippets")
    monkeypatch.setattr(config, "MEDIA_ROOT", tmp_path / "media")
    monkeypatch.setattr(lernen, "bestand", lernen.Bestand())
    monkeypatch.setattr(lernen, "trainieren_im_hintergrund", lambda: None)
    (tmp_path / "media").mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=150", "-b:a", "128k",
                    str(tmp_path / "media" / "ton.mp3")], check=True)
    con = db.verbinden()
    lid = con.execute("INSERT INTO lieder(pfad, status, dauer, snippet_start) VALUES ('ton.mp3', 'analysiert', 150, 20)").lastrowid
    gid = con.execute("INSERT INTO genres(name) VALUES ('Test')").lastrowid
    con.commit()
    with TestClient(web.app) as client:
        yield client, con, lid, gid


def test_bewerten_nur_mit_bestehendem_genre(umgebung):
    client, con, lid, gid = umgebung
    assert client.post("/api/bewerten", json={"lied_id": lid, "haupt": 9999}).status_code == 409
    assert client.post("/api/bewerten", json={"lied_id": lid, "haupt": gid, "tags": [9999]}).status_code == 409
    assert client.post("/api/bewerten", json={"lied_id": 9999, "haupt": gid}).status_code == 404
    assert con.execute("SELECT COUNT(*) FROM etiketten").fetchone()[0] == 0
    assert client.post("/api/bewerten", json={"lied_id": lid, "haupt": gid}).status_code == 200
    assert con.execute("SELECT COUNT(*) FROM etiketten").fetchone()[0] == 1


def test_snippet_parallel_ohne_fehler(umgebung):
    client, con, lid, gid = umgebung
    with ThreadPoolExecutor(6) as pool:
        codes = list(pool.map(lambda _: client.get(f"/api/snippet/{lid}").status_code, range(6)))
    assert codes == [200] * 6
    reste = [p.name for p in config.SNIPPETS.iterdir() if p.name.endswith(".tmp.mp3")]
    assert reste == []
    r = client.get(f"/api/snippet/{lid}", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_duplikat_entscheidung_einmal(umgebung):
    client, con, lid, gid = umgebung
    lid2 = con.execute("INSERT INTO lieder(pfad, status) VALUES ('kopie.mp3', 'analysiert')").lastrowid
    grp = con.execute("INSERT INTO dup_gruppen(erstellt) VALUES (0)").lastrowid
    con.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,?)",
                    [(grp, lid, 1), (grp, lid2, 0)])
    con.commit()
    m = [lid, lid2]
    assert client.post(f"/api/duplikate/{grp}", json={"status": "freigegeben", "behalten": 12345, "mitglieder": m}).status_code == 400
    # App zeigte eine andere Zusammensetzung: ablehnen
    assert client.post(f"/api/duplikate/{grp}", json={"status": "freigegeben", "behalten": lid2, "mitglieder": [lid]}).status_code == 409
    assert client.post(f"/api/duplikate/{grp}", json={"status": "freigegeben", "behalten": lid2, "mitglieder": m}).status_code == 200
    assert client.post(f"/api/duplikate/{grp}", json={"status": "verworfen", "mitglieder": m}).status_code == 404  # schon entschieden
    behalten = con.execute("SELECT lied_id FROM dup_mitglieder WHERE gruppe_id=? AND behalten=1", (grp,)).fetchall()
    assert [r[0] for r in behalten] == [lid2]


def test_snippet_pfad_bleibt_in_media(umgebung):
    client, con, lid, gid = umgebung
    boese = con.execute("INSERT INTO lieder(pfad, status, dauer) VALUES ('../../etc/passwd', 'analysiert', 100)").lastrowid
    con.commit()
    assert client.get(f"/api/snippet/{boese}").status_code in (400, 404)


def test_genre_anlegen_ohne_schreibweisen_dublette(umgebung):
    client, con, lid, gid = umgebung
    a = client.post("/api/genres", json={"name": "Hard Techno"}).json()["id"]
    assert client.post("/api/genres", json={"name": "hard-techno"}).json()["id"] == a
    assert client.post("/api/genres", json={"name": "Hard Techno", "eltern_id": a}).json()["id"] != a  # anderes Elternteil: erlaubt
    assert client.post("/api/genres", json={"name": "X", "eltern_id": 9999}).status_code == 400


def test_duplikat_entscheidung_zuruecknehmen(umgebung):
    client, con, lid, gid = umgebung
    lid2 = con.execute("INSERT INTO lieder(pfad, status) VALUES ('k.mp3', 'analysiert')").lastrowid
    grp = con.execute("INSERT INTO dup_gruppen(erstellt) VALUES (0)").lastrowid
    con.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,?)", [(grp, lid, 1), (grp, lid2, 0)])
    con.commit()
    assert client.post(f"/api/duplikate/{grp}/zuruecknehmen").status_code == 404   # noch offen
    client.post(f"/api/duplikate/{grp}", json={"status": "freigegeben", "behalten": lid, "mitglieder": [lid, lid2]})
    assert client.post(f"/api/duplikate/{grp}/zuruecknehmen").status_code == 200
    assert con.execute("SELECT status FROM dup_gruppen WHERE id=?", (grp,)).fetchone()[0] == "offen"


def test_stand_liefert_ziel_und_quote(umgebung):
    client, con, lid, gid = umgebung
    s = client.get("/api/stand").json()
    assert s["ziel"] == config.ZIEL_TREFFERQUOTE and s["min_auto"] == config.MIN_BEWERTUNGEN_AUTO


def test_zugang_nur_mit_schluessel(umgebung, monkeypatch):
    client, con, lid, gid = umgebung
    monkeypatch.setattr(web, "SCHLUESSEL", "geheim123")
    client.cookies.clear()
    assert client.get("/api/genres").status_code == 401
    assert client.get("/").status_code == 401
    assert client.get("/?schluessel=falsch", follow_redirects=False).status_code == 401
    r = client.get("/?schluessel=geheim123", follow_redirects=False)
    assert r.status_code == 303 and "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/api/genres").status_code == 200   # Cookie gesetzt
    assert client.get("/api/genres").headers["x-frame-options"] == "DENY"


def test_grenzen(umgebung):
    client, con, lid, gid = umgebung
    assert client.get("/api/naechste?anzahl=1000").status_code == 422
    assert client.get(f"/api/aehnliche/{lid}?n=10000").status_code == 422
    assert client.get("/api/aehnliche/999999").status_code == 404
    assert client.get(f"/api/snippet/{lid}?ausschnitt=99").status_code == 422
    gross = client.post("/api/bewerten", content=b"{" + b" " * 70000 + b"}", headers={"Content-Type": "application/json"})
    assert gross.status_code == 413
    assert client.post("/api/bewerten", json={"lied_id": lid, "haupt": gid, "tags": list(range(50))}).status_code == 422


def test_entscheidungen_werden_protokolliert(umgebung):
    client, con, lid, gid = umgebung
    lid2 = con.execute("INSERT INTO lieder(pfad, status) VALUES ('k.mp3', 'analysiert')").lastrowid
    grp = con.execute("INSERT INTO dup_gruppen(erstellt) VALUES (0)").lastrowid
    con.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,?)", [(grp, lid, 1), (grp, lid2, 0)])
    con.commit()
    client.post(f"/api/duplikate/{grp}", json={"status": "freigegeben", "behalten": lid, "mitglieder": [lid, lid2]})
    assert con.execute("SELECT aktion FROM protokoll").fetchall()[-1][0] == "duplikat_freigegeben"


def test_cache_aufraeumen_loescht_aelteste(umgebung, tmp_path):
    client, con, lid, gid = umgebung
    import os
    config.SNIPPETS.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        f = config.SNIPPETS / f"{i}_0.mp3"
        f.write_bytes(b"x" * 1000)
        os.utime(f, (i, i))
    web.cache_aufraeumen(max_bytes=5000)
    uebrig = sorted(int(f.name.split("_")[0]) for f in config.SNIPPETS.glob("*.mp3"))
    assert uebrig == [6, 7, 8, 9]   # bis 80 % der Grenze, älteste zuerst weg


def _gruppe(con, mitglieder, behalten, sicher=1):
    g = con.execute("INSERT INTO dup_gruppen(erstellt, sicher) VALUES (0, ?)", (sicher,)).lastrowid
    con.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,?)",
                    [(g, m, int(m == behalten)) for m in mitglieder])
    return g


def test_freigaben_koennen_keine_fassung_ganz_verlieren(umgebung):
    """X Original, Y Kopie, Z Radio-Edit: {X,Y} sicher, {X,Z} Zweifel. Nie dürfen X und Y beide weg."""
    client, con, x, gid = umgebung
    y = con.execute("INSERT INTO lieder(pfad, status) VALUES ('y.mp3', 'analysiert')").lastrowid
    z = con.execute("INSERT INTO lieder(pfad, status) VALUES ('z_edit.mp3', 'analysiert')").lastrowid
    g1 = _gruppe(con, [x, y], x)
    g2 = _gruppe(con, [x, z], x, sicher=0)
    con.commit()
    assert client.post(f"/api/duplikate/{g1}", json={"status": "freigegeben", "mitglieder": [x, y]}).status_code == 200
    r = client.post(f"/api/duplikate/{g2}", json={"status": "freigegeben", "behalten": z, "mitglieder": [x, z]})
    assert r.status_code == 409 and "anderen Gruppe" in r.json()["detail"]
    # „alle behalten“ geht immer
    assert client.post(f"/api/duplikate/{g2}", json={"status": "verworfen", "mitglieder": [x, z]}).status_code == 200


def test_freigabe_nur_wenn_alle_dateien_aktuell(umgebung):
    client, con, x, gid = umgebung
    y = con.execute("INSERT INTO lieder(pfad, status) VALUES ('y.mp3', 'fehlt')").lastrowid
    g = _gruppe(con, [x, y], x)
    con.commit()
    r = client.post(f"/api/duplikate/{g}", json={"status": "freigegeben", "mitglieder": [x, y]})
    assert r.status_code == 409


def test_ruecknahme_nach_neuberechnung_ohne_doppelte_gruppe(umgebung):
    client, con, a, gid = umgebung
    b = con.execute("INSERT INTO lieder(pfad, status) VALUES ('b.mp3', 'analysiert')").lastrowid
    c = con.execute("INSERT INTO lieder(pfad, status) VALUES ('c.mp3', 'analysiert')").lastrowid
    g1 = _gruppe(con, [a, b], a)
    con.commit()
    client.post(f"/api/duplikate/{g1}", json={"status": "freigegeben", "mitglieder": [a, b]})
    _gruppe(con, [a, c], a)   # inzwischen neu gerechnet: A steckt in einer neuen offenen Gruppe
    con.commit()
    r = client.post(f"/api/duplikate/{g1}/zuruecknehmen").json()
    assert r["neu_vorgelegt"] is True
    doppelt = con.execute("SELECT m.lied_id FROM dup_mitglieder m JOIN dup_gruppen g ON g.id=m.gruppe_id "
                          "WHERE g.status='offen' GROUP BY m.lied_id HAVING COUNT(*) > 1").fetchall()
    assert doppelt == []


def test_gescheitertes_umbenennen_sperrt_die_datenbank_nicht(umgebung):
    """Prüfrunde 3B: Umbenennen auf vorhandenen Namen hielt die Schreibsperre fest → App hing."""
    client, con, lid, gid = umgebung
    pop = client.post("/api/genres", json={"name": "Pop"}).json()["id"]
    blues = client.post("/api/genres", json={"name": "Blues"}).json()["id"]
    r = client.patch(f"/api/genres/{blues}", json={"name": "pop"})
    assert r.status_code == 409 and "Zusammenführen" in r.json()["detail"]   # auch Groß/Klein zählt
    import sqlite3
    pruef = sqlite3.connect(config.DB_PFAD, timeout=1)
    pruef.execute("BEGIN IMMEDIATE")   # würde mit „database is locked“ scheitern, wenn die Sperre hinge
    pruef.rollback()
    assert client.post("/api/bewerten", json={"lied_id": lid, "haupt": pop}).status_code == 200


def test_gleichzeitige_genre_aenderungen_bleiben_konsistent(umgebung):
    """Prüfrunde 4: paralleles Zusammenführen A→B und B→C, paralleles Anlegen „Rock“/„rock“."""
    from concurrent.futures import ThreadPoolExecutor
    client, con, lid, gid = umgebung
    for runde in range(8):
        a = client.post("/api/genres", json={"name": f"A{runde}"}).json()["id"]
        b = client.post("/api/genres", json={"name": f"B{runde}"}).json()["id"]
        c = client.post("/api/genres", json={"name": f"C{runde}"}).json()["id"]
        client.post("/api/bewerten", json={"lied_id": lid, "haupt": a})
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda qz: client.post(f"/api/genres/{qz[0]}/zusammenfuehren", json={"ziel_id": qz[1]}),
                          [(a, b), (b, c)]))
        verwaist = con.execute("SELECT COUNT(*) FROM etiketten WHERE genre_id NOT IN (SELECT id FROM genres)").fetchone()[0]
        assert verwaist == 0
    with ThreadPoolExecutor(8) as pool:
        ids = set(pool.map(lambda n: client.post("/api/genres", json={"name": n}).json()["id"],
                           ["Rock", "rock", "ROCK", "Rock ", " rock", "Rock", "rOck", "ROCK"]))
    assert len(ids) == 1


def test_riesige_zahlen_sauber_abgelehnt(umgebung):
    client, con, lid, gid = umgebung
    for r in (client.post("/api/ueberspringen", json={"lied_id": 2**70}),
              client.get(f"/api/aehnliche/{2**64}"),
              client.post("/api/bewerten", json={"lied_id": 2**64, "haupt": gid})):
        assert r.status_code in (400, 404, 422)
    assert client.get("/api/genres").status_code == 200   # Verbindung danach in Ordnung


def test_erledigte_duplikatfrage_wird_ausgeblendet(umgebung):
    client, con, a, gid = umgebung
    b = con.execute("INSERT INTO lieder(pfad, status) VALUES ('b.mp3', 'analysiert')").lastrowid
    c = con.execute("INSERT INTO lieder(pfad, status) VALUES ('c.mp3', 'analysiert')").lastrowid
    p1 = _gruppe(con, [a, b], b, sicher=0)
    p2 = _gruppe(con, [a, c], c, sicher=0)
    con.commit()
    assert client.post(f"/api/duplikate/{p2}", json={"status": "freigegeben", "mitglieder": [a, c]}).status_code == 200
    assert [g["id"] for g in client.get("/api/duplikate").json()] == []          # P1 ist erledigt (a ist weg)
    assert client.post(f"/api/duplikate/{p1}", json={"status": "verworfen", "mitglieder": [a, b]}).status_code == 409


def test_analyse_status(umgebung, monkeypatch, tmp_path):
    client, con, lid, gid = umgebung
    jetzt = db.jetzt()
    for i, alter in enumerate([60, 3700, 3800, 4 * 3600]):
        con.execute("INSERT INTO lieder(pfad, status, analysiert, analyse_s) VALUES (?, 'analysiert', ?, 10)",
                    (f"a{i}.mp3", jetzt - alter))
    con.execute("INSERT INTO lieder(pfad, status) VALUES ('offen.mp3', 'neu')")
    con.commit()
    monkeypatch.setattr(web, "ANALYSE_CGROUP", tmp_path / "gibtsnicht")
    s = client.get("/api/analyse?stunden=6").json()
    assert s["lieder"] == {"analysiert": 5, "neu": 1}
    assert len(s["verlauf"]) == 6 and sum(v["n"] for v in s["verlauf"]) == 4
    assert s["letzte_stunde"] == 1 and s["tempo_n"] >= 1 and s["tempo"] % 4 == 0 and s["tempo_min"] <= s["tempo"] <= s["tempo_max"]
    assert s["dienst"] == {"laeuft": False}
    cg = tmp_path / "cg"
    cg.mkdir()
    (cg / "cgroup.procs").write_text("123\n")
    (cg / "memory.current").write_text(str(2 * 1048576 * 1024))
    (cg / "memory.high").write_text(str(2500 * 1048576))
    (cg / "memory.max").write_text("max")
    (cg / "cpu.stat").write_text("nr_periods 10\nnr_throttled 5\n")
    monkeypatch.setattr(web, "ANALYSE_CGROUP", cg)
    d = client.get("/api/analyse").json()["dienst"]
    assert d["laeuft"] and d["speicher_mb"] == 2048 and d["bremse_mb"] == 2500 and d["grenze_mb"] is None
    assert d["cpu_gebremst"] == 0.5
    assert client.get("/status").status_code == 200 and client.get("/tokens.css").status_code == 200
    assert client.get("/api/analyse?stunden=1").status_code == 422


def test_genre_mit_neuem_obergenre(umgebung):
    client, con, lid, gid = umgebung
    kind = client.post("/api/genres", json={"name": "Disco House", "eltern_name": "House"}).json()["id"]
    ober = con.execute("SELECT eltern_id FROM genres WHERE id=?", (kind,)).fetchone()[0]
    assert tuple(con.execute("SELECT name, eltern_id FROM genres WHERE id=?", (ober,)).fetchone()) == ("House", None)
    # gibt es das Obergenre schon (andere Schreibweise), wird es genommen statt doppelt angelegt
    zwei = client.post("/api/genres", json={"name": "Deep House", "eltern_name": "house"}).json()["id"]
    assert con.execute("SELECT eltern_id FROM genres WHERE id=?", (zwei,)).fetchone()[0] == ober
    assert client.post("/api/genres", json={"name": "Disco House", "eltern_name": "House"}).json()["id"] == kind
    # ungültiger Obergenre-Name: nichts wird angelegt
    vorher = con.execute("SELECT COUNT(*) FROM genres").fetchone()[0]
    assert client.post("/api/genres", json={"name": "Acid", "eltern_name": "A/B"}).status_code == 400
    assert client.post("/api/genres", json={"name": "Acid", "eltern_name": "X", "eltern_id": ober}).status_code == 400
    assert con.execute("SELECT COUNT(*) FROM genres").fetchone()[0] == vorher


def test_stand_noetige_mengen(umgebung):
    client, con, lid, gid = umgebung
    s = client.get("/api/stand").json()
    assert s["bewertungen_mind"] == 32  # 95 % vorsichtig (z = 1,28) erst ab 32 fehlerfreien Treffern
    assert web._stichproben_noetig(0, 0) == 15 and web._stichproben_noetig(9, 10) == 32
    assert web.lernen.wilson_unten(15, 15) >= 0.90 > web.lernen.wilson_unten(14, 14)


def test_ganze_datei_mit_springen(umgebung):
    """Scrub-Leiste: ganze Originaldatei, Teilstücke (Range), nur unter der Freigabe, nur abspielbare Formate."""
    client, con, lid, gid = umgebung
    con.execute("UPDATE lieder SET codec='mp3' WHERE id=?", (lid,))
    ok = lid
    wma = con.execute("INSERT INTO lieder(pfad, status, codec) VALUES ('alt.wma', 'analysiert', 'wmav2')").lastrowid
    raus = con.execute("INSERT INTO lieder(pfad, status, codec) VALUES ('../geheim.mp3', 'analysiert', 'mp3')").lastrowid
    weg = con.execute("INSERT INTO lieder(pfad, status, codec) VALUES ('fehlt.mp3', 'analysiert', 'mp3')").lastrowid
    con.commit()
    voll = client.get(f"/api/datei/{ok}")
    assert voll.status_code == 200 and voll.headers["content-type"] == "audio/mpeg"
    assert voll.headers.get("accept-ranges") == "bytes"
    teil = client.get(f"/api/datei/{ok}", headers={"Range": "bytes=0-99"})
    assert teil.status_code == 206 and len(teil.content) == 100
    assert client.get(f"/api/datei/{wma}").status_code == 415
    assert client.get(f"/api/datei/{raus}").status_code == 400
    assert client.get(f"/api/datei/{weg}").status_code == 404
    assert client.get("/api/datei/99999").status_code == 404
    assert web._lied(con, ok)["direkt"] is True and web._lied(con, wma)["direkt"] is False


def test_geraete_protokoll_landet_im_journal(umgebung, capsys):
    client, *_ = umgebung
    r = client.post("/api/protokoll", json={"ereignis": "sprung", "ok": False, "ziel": 120, "springbar": [[0, 30]]},
                    headers={"User-Agent": "iPhone Test"})
    assert r.status_code == 200
    aus = capsys.readouterr().out
    assert "Gerät:" in aus and '"ziel": 120' in aus and "iPhone Test" in aus
    assert client.post("/api/protokoll", json={"ereignis": "x" * 41}).status_code == 422


def test_fortschritt_zaehlt_abdeckung_und_duplikate(umgebung):
    client, con, lid, gid = umgebung
    lid2 = con.execute("INSERT INTO lieder(pfad, status) VALUES ('b.mp3', 'neu')").lastrowid
    con.execute("INSERT INTO lieder(pfad, status) VALUES ('weg.mp3', 'ausgeschlossen')")
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (lid, gid))
    g = con.execute("INSERT INTO dup_gruppen(erstellt, status) VALUES (0, 'verworfen')").lastrowid
    con.execute("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,1)", (g, lid))
    con.commit()
    f = client.get("/api/fortschritt").json()
    assert f["analyse"]["gesamt"] == 2 and f["analyse"]["fertig"] == 1 and f["analyse"]["rest"] == 1  # ohne ausgeschlossene
    assert f["genres"]["bewertet"] == 1 and f["genres"]["abgedeckt"] == 1
    assert f["genres"]["naechstes"]["fehlt"] == config.MIN_BEWERTUNGEN_AUTO - 1
    assert f["duplikate"] == {"offen": 0, "entschieden": 1}


def test_duplikate_empfehlung_beste_qualitaet_oben(umgebung):
    client, con, lid, gid = umgebung
    a = con.execute("INSERT INTO lieder(pfad, status, codec, bitrate, verlustfrei) VALUES ('a.mp3','analysiert','mp3',128,0)").lastrowid
    b = con.execute("INSERT INTO lieder(pfad, status, codec, bitrate, verlustfrei) VALUES ('b.mp3','analysiert','mp3',192,0)").lastrowid
    g = _gruppe(con, [a, b], a)   # gespeicherte Wahl (z. B. nach Rücknahme) ist die schlechtere
    con.commit()
    m = client.get("/api/duplikate").json()[0]["mitglieder"]
    assert m[0]["id"] == b and m[0]["empfohlen"] and not m[1]["empfohlen"]


def test_automatik_uebersicht_und_zuruecknehmen(umgebung):
    client, con, a, gid = umgebung
    b = con.execute("INSERT INTO lieder(pfad, status, codec, bitrate, verlustfrei) VALUES ('b.mp3','analysiert','mp3',320,0)").lastrowid
    g = _gruppe(con, [a, b], a)
    con.commit()
    from musiksort import duplikate
    con.execute("BEGIN IMMEDIATE")
    duplikate.entscheiden(con, g, "freigegeben", [a, b], b, von="auto:2", grund="eng")
    con.commit()
    assert client.get("/api/duplikate-auto").json()["stufen"] == {"2": {"freigegeben": 1}}
    assert client.post("/api/duplikate-auto/zuruecknehmen", json={"stufe": "X"}).status_code == 422
    assert client.post("/api/duplikate-auto/zuruecknehmen", json={}).json() == {"zurueckgenommen": 1}
    assert con.execute("SELECT status FROM dup_gruppen WHERE id=?", (g,)).fetchone()[0] == "offen"
    assert con.execute("SELECT aktion FROM protokoll ORDER BY id DESC").fetchone()[0] == "duplikat_zuruecknehmen"


def test_stichprobe_meldet_richtig_oder_falsch(umgebung):
    client, con, lid, gid = umgebung
    unter = con.execute("INSERT INTO genres(name, eltern_id) VALUES ('Unter', ?)", (gid,)).lastrowid
    anders = con.execute("INSERT INTO genres(name) VALUES ('Anders')").lastrowid
    con.commit()
    # Wahl im Teilbaum des geprüften Genres zählt als richtig
    r = client.post("/api/bewerten", json={"lied_id": lid, "haupt": unter, "stichprobe": True, "vorschlag_id": gid}).json()
    assert r["stichprobe"] == {"richtig": True, "modell": "Test"}
    r = client.post("/api/bewerten", json={"lied_id": lid, "haupt": anders, "stichprobe": True, "vorschlag_id": gid}).json()
    assert r["stichprobe"]["richtig"] is False
    assert client.post("/api/bewerten", json={"lied_id": lid, "haupt": gid}).json()["stichprobe"] is None


def test_zurueck_stellt_vorherigen_stand_her(umgebung):
    client, con, lid, gid = umgebung
    anders = con.execute("INSERT INTO genres(name) VALUES ('Anders')").lastrowid
    con.commit()
    erst = client.post("/api/bewerten", json={"lied_id": lid, "haupt": gid}).json()["rueck_id"]
    zweit = client.post("/api/bewerten", json={"lied_id": lid, "haupt": anders, "stichprobe": True,
                                               "vorschlag_id": gid}).json()["rueck_id"]
    assert con.execute("SELECT COUNT(*) FROM stichproben").fetchone()[0] == 1
    assert client.post("/api/rueckgaengig", json={"rueck_id": zweit}).json()["lied_id"] == lid
    # erste Bewertung wieder da, Stichprobe weg
    assert [r[0] for r in con.execute("SELECT genre_id FROM etiketten WHERE lied_id=?", (lid,))] == [gid]
    assert con.execute("SELECT COUNT(*) FROM stichproben").fetchone()[0] == 0
    assert client.post("/api/rueckgaengig", json={"rueck_id": zweit}).status_code == 404  # nur einmal
    client.post("/api/rueckgaengig", json={"rueck_id": erst})
    assert con.execute("SELECT COUNT(*) FROM etiketten").fetchone()[0] == 0


def test_zurueck_nach_spaeter(umgebung):
    client, con, lid, gid = umgebung
    rid = client.post("/api/ueberspringen", json={"lied_id": lid}).json()["rueck_id"]
    assert con.execute("SELECT COUNT(*) FROM uebersprungen").fetchone()[0] == 1
    client.post("/api/rueckgaengig", json={"rueck_id": rid})
    assert con.execute("SELECT COUNT(*) FROM uebersprungen").fetchone()[0] == 0
