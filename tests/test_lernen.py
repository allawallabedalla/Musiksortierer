"""Lernschleife und Duplikate mit künstlichen Daten (kein Audio, kein NAS)."""
import numpy as np
import pytest

from musiksort import config, db, duplikate, lernen


@pytest.fixture
def con(tmp_path, monkeypatch):
    monkeypatch.setattr(lernen, "bestand", lernen.Bestand())
    return db.verbinden(tmp_path / "t.db")


def _wolken(con, n_pro=120, rauschen=0.6):
    """3 klar getrennte Klang-Wolken, je n_pro Lieder."""
    rng = np.random.default_rng(1)
    zentren = rng.normal(size=(3, 1280))
    wahr = {}
    for w in range(3):
        for _ in range(n_pro):
            v = zentren[w] + rauschen * rng.normal(size=1280)
            v = (v / np.linalg.norm(v)).astype(np.float16)
            lid = con.execute("INSERT INTO lieder(pfad, status, emb) VALUES (?, 'analysiert', ?)",
                              (f"{w}/{len(wahr)}.mp3", v.tobytes())).lastrowid
            wahr[lid] = w
    gids = [con.execute("INSERT INTO genres(name) VALUES (?)", (f"G{w}",)).lastrowid for w in range(3)]
    con.commit()
    return wahr, gids


def _bewerte(con, lid, gid):
    con.execute("INSERT INTO etiketten VALUES (?,?,1,'ich',0)", (lid, gid))


def test_sortiert_selbst_erst_mit_genug_beispielen(con):
    wahr, gids = _wolken(con)
    je = {w: [l for l, x in wahr.items() if x == w] for w in range(3)}
    for w in range(3):
        for lid in je[w][:20]:
            _bewerte(con, lid, gids[w])
    lernen.trainieren(con)
    assert con.execute("SELECT COUNT(*) FROM vorhersagen WHERE auto=1").fetchone()[0] == 0

    for w in range(3):
        for lid in je[w][20:40]:
            _bewerte(con, lid, gids[w])
    lernen.trainieren(con)
    auto = con.execute("SELECT lied_id, genre_id FROM vorhersagen WHERE auto=1").fetchall()
    assert len(auto) > 200
    assert sum(gids[wahr[l]] != g for l, g in auto) / len(auto) < 0.05
    # selbst bewertete Lieder sind nie „automatisch“
    bewertet = {r[0] for r in con.execute("SELECT lied_id FROM etiketten")}
    assert not bewertet & {l for l, _ in auto}


def test_backlog_ohne_bewertete(con):
    wahr, gids = _wolken(con, n_pro=30)
    erste = next(iter(wahr))
    _bewerte(con, erste, gids[0])
    con.commit()
    karten = lernen.naechste(con, 5)
    assert len(karten) == 5 and erste not in {k["lied_id"] for k in karten}


def test_wilson_vorsichtiger_als_rohquote():
    assert lernen.wilson_unten(31, 31) < 0.95 <= lernen.wilson_unten(32, 32)
    assert lernen.wilson_oben(5, 10) < 0.95 < lernen.wilson_oben(10, 10)


def test_schwelle_nur_wenn_belegbar():
    p = np.linspace(0.99, 0.5, 200)
    richtig = np.r_[np.ones(150, bool), np.zeros(50, bool)]
    s = lernen.schwelle_finden(p, richtig)
    assert s is not None and s[0] > 0.6 and s[1] >= config.ZIEL_TREFFERQUOTE
    assert lernen.schwelle_finden(p[:20], richtig[:20]) is None  # 20/20 reicht nicht


def test_duplikate_sicher_und_zweifel(con):
    rng = np.random.default_rng(3)
    fp = rng.integers(0, 2**32, 900, dtype=np.uint64).astype(np.uint32)
    anders = rng.integers(0, 2**32, 900, dtype=np.uint64).astype(np.uint32)
    gekippt = fp.copy()
    gekippt[::5] ^= np.uint32(0xFFFF)  # ~10 % Bitfehler: wie ein Edit mit leicht anderem Mastering

    def lied(pfad, f, dauer, verlustfrei, bitrate):
        con.execute("INSERT INTO lieder(pfad, status, fp, dauer, verlustfrei, bitrate) "
                    "VALUES (?, 'analysiert', ?, ?, ?, ?)", (pfad, f.tobytes(), dauer, verlustfrei, bitrate))
    lied("a.flac", fp, 300, 1, 900)
    lied("a.mp3", fp, 300.5, 0, 320)
    lied("a_radio_edit.mp3", gekippt, 290, 0, 320)   # 10 s kürzer: Zweifel
    lied("fremd.mp3", anders, 300, 0, 320)
    con.commit()
    duplikate.finden(con)
    gruppen = con.execute("SELECT id, sicher FROM dup_gruppen").fetchall()
    sicher = [g for g in gruppen if g["sicher"]]
    zweifel = [g for g in gruppen if not g["sicher"]]
    assert len(sicher) == 1 and len(zweifel) == 1
    behalten = con.execute("SELECT l.pfad FROM dup_mitglieder m JOIN lieder l ON l.id=m.lied_id "
                           "WHERE m.gruppe_id=? AND m.behalten=1", (sicher[0]["id"],)).fetchone()[0]
    assert behalten == "a.flac"
    im_zweifel = {r[0] for r in con.execute("SELECT l.pfad FROM dup_mitglieder m JOIN lieder l ON l.id=m.lied_id "
                                             "WHERE m.gruppe_id=?", (zweifel[0]["id"],))}
    assert im_zweifel == {"a.flac", "a_radio_edit.mp3"}


def test_env_behaelt_anfuehrungszeichen_im_wert(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    (tmp_path / ".env").write_text("A=geheim'\nB='umschlossen'\n")
    assert config.env("A") == "geheim'"
    assert config.env("B") == "umschlossen"


def _fp_lied(con, pfad, fp, dauer, verlustfrei=0, bitrate=320):
    return con.execute("INSERT INTO lieder(pfad, status, fp, dauer, verlustfrei, bitrate) VALUES (?, 'analysiert', ?, ?, ?, ?)",
                       (pfad, fp.tobytes(), dauer, verlustfrei, bitrate)).lastrowid


def test_duplikate_ids_nie_wiederverwendet_und_entschiedenes_bleibt(con):
    rng = np.random.default_rng(5)
    fp = rng.integers(0, 2**32, 900, dtype=np.uint64).astype(np.uint32)
    a = _fp_lied(con, "a.flac", fp, 300, 1, 900)
    b = _fp_lied(con, "b.mp3", fp, 300)
    con.commit()
    duplikate.finden(con)
    g1 = con.execute("SELECT id FROM dup_gruppen WHERE status='offen'").fetchone()[0]
    c = _fp_lied(con, "c.mp3", fp, 300.2)
    con.commit()
    duplikate.finden(con)   # neu berechnet: alte Gruppe ersetzt, neue ID
    g2 = con.execute("SELECT id FROM dup_gruppen WHERE status='offen'").fetchone()[0]
    assert g2 != g1 and con.execute("SELECT status FROM dup_gruppen WHERE id=?", (g1,)).fetchone()[0] == "ersetzt"
    # freigeben: b und c werden gelöscht; danach darf nichts abstürzen oder wiederkommen
    con.execute("UPDATE dup_gruppen SET status='freigegeben' WHERE id=?", (g2,))
    d = _fp_lied(con, "d_edit.mp3", fp, 312)   # Zweifelsfall zu einem entschiedenen Lied
    con.commit()
    duplikate.finden(con)
    offen = con.execute("SELECT g.sicher, group_concat(l.pfad) FROM dup_gruppen g JOIN dup_mitglieder m ON m.gruppe_id=g.id "
                        "JOIN lieder l ON l.id=m.lied_id WHERE g.status='offen' GROUP BY g.id").fetchall()
    assert [tuple(r) for r in offen] == [(0, "a.flac,d_edit.mp3")]
    assert con.execute("SELECT COUNT(*) FROM dup_mitglieder m JOIN dup_gruppen g ON g.id=m.gruppe_id WHERE g.status='offen' AND m.lied_id IN (?,?)", (b, c)).fetchone()[0] == 0  # zu Löschendes kommt nicht wieder


def test_duplikate_keine_ketten(con):
    rng = np.random.default_rng(6)
    fp = rng.integers(0, 2**32, 900, dtype=np.uint64).astype(np.uint32)
    for i, dauer in enumerate([300, 302.4, 304.8, 307.2]):   # je 2,4 s Abstand, a–d aber 7,2 s
        _fp_lied(con, f"{i}.mp3", fp, dauer, 1 if i == 0 else 0)
    con.commit()
    duplikate.finden(con)
    sicher = con.execute("SELECT group_concat(l.pfad) FROM dup_gruppen g JOIN dup_mitglieder m ON m.gruppe_id=g.id "
                         "JOIN lieder l ON l.id=m.lied_id WHERE g.status='offen' AND g.sicher=1 GROUP BY g.id").fetchall()
    assert [r[0] for r in sicher] == ["0.mp3,1.mp3"]   # nur, was direkt zur behaltenen Datei passt


def test_unbekannte_genres_werden_nicht_zwangszugeordnet(con):
    """3 bewertete Wolken + 3 benachbarte Wolken ohne Genre: die fremden dürfen nicht „automatisch“ werden."""
    rng = np.random.default_rng(7)
    zentren = rng.normal(size=(6, 1280))
    wahr = {}
    for w in range(6):
        for _ in range(80):
            v = zentren[w] + 0.6 * rng.normal(size=1280)
            v = (v / np.linalg.norm(v)).astype(np.float16)
            wahr[con.execute("INSERT INTO lieder(pfad, status, emb) VALUES (?, 'analysiert', ?)",
                             (f"{w}/{len(wahr)}.mp3", v.tobytes())).lastrowid] = w
    gids = [con.execute("INSERT INTO genres(name) VALUES (?)", (f"G{w}",)).lastrowid for w in range(3)]
    for w in range(3):
        for lid in [l for l, x in wahr.items() if x == w][:40]:
            _bewerte(con, lid, gids[w])
    con.commit()
    lernen.trainieren(con)
    auto = con.execute("SELECT lied_id, genre_id FROM vorhersagen WHERE auto=1").fetchall()
    fremd = sum(wahr[l] >= 3 for l, _ in auto)
    assert len(auto) > 80 and fremd / len(auto) < 0.05


def test_backlog_klanggruppen(con):
    wahr, gids = _wolken(con, n_pro=60)
    lernen.cluster_berechnen(con, pro_gruppe=60)
    gruppen = dict(con.execute("SELECT lied_id, gruppe FROM klanggruppen").fetchall())
    for w in range(3):  # jede Wolke landet geschlossen in einer Gruppe
        assert len({gruppen[l] for l, x in wahr.items() if x == w}) == 1
    karten = lernen.naechste(con, 3)
    assert [k["grund"] for k in karten] == ["gruppe"] * 3
    assert len({gruppen[k["lied_id"]] for k in karten}) == 3   # drei verschiedene Gruppen


def test_stichproben_belasten_das_vorhergesagte_genre(con):
    wahr, gids = _wolken(con)
    je = {w: [l for l, x in wahr.items() if x == w] for w in range(3)}
    for w in range(3):
        for lid in je[w][:40]:
            _bewerte(con, lid, gids[w])
    con.commit()
    lernen.trainieren(con)
    assert con.execute("SELECT COUNT(*) FROM modell_stand WHERE schwelle IS NOT NULL").fetchone()[0] == 3
    # 12 Stichproben: Modell sagte G0, du hast G1 gewählt → G0 ist widerlegt, G1 bleibt
    con.executemany("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)",
                    [(je[0][50 + i], gids[0], gids[1]) for i in range(12)])
    # 15 richtige für G2 → positiv belegt
    con.executemany("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)",
                    [(je[2][50 + i], gids[2], gids[2]) for i in range(15)])
    con.commit()
    lernen.trainieren(con)
    stand = {r["genre_id"]: r for r in con.execute("SELECT * FROM modell_stand")}
    assert stand[gids[0]]["schwelle"] is None                   # abgeschaltet
    assert stand[gids[1]]["schwelle"] is not None and not stand[gids[1]]["belegt"]
    assert stand[gids[2]]["belegt"] == 1


def test_kleines_genre_zaehlt_als_fehler_statt_2fach_cv(con):
    wahr, gids = _wolken(con)
    je = {w: [l for l, x in wahr.items() if x == w] for w in range(3)}
    for w in range(2):
        for lid in je[w][:40]:
            _bewerte(con, lid, gids[w])
    for lid in je[2][:3]:   # nur 3 Beispiele: nicht in der Kreuzvalidierung, aber als Prüfstein
        _bewerte(con, lid, gids[2])
    con.commit()
    y = np.array([gids[0]] * 40 + [gids[1]] * 40 + [gids[2]] * 3)
    X = np.stack([np.frombuffer(con.execute("SELECT emb FROM lieder WHERE id=?", (l,)).fetchone()[0], np.float16)
                  for l in je[0][:40] + je[1][:40] + je[2][:3]]).astype(np.float32)
    wahr_cv, vorh, _ = lernen._kreuzvalidierung(X, y, {gids[0]: 40, gids[1]: 40, gids[2]: 3})[:3]
    assert len(wahr_cv) == 83 and gids[2] not in set(vorh)   # kleine Klasse wird nie vorhergesagt → zählt falsch


def test_backlog_sucht_mehr_beispiele_fuer_haeufige_genres(con):
    wahr, gids = _wolken(con, n_pro=120)
    je = {w: [l for l, x in wahr.items() if x == w] for w in range(3)}
    for w in range(3):
        for lid in je[w][:5]:
            _bewerte(con, lid, gids[w])
    con.commit()
    lernen.trainieren(con)
    gruende = [k["grund"] for _ in range(10) for k in lernen.naechste(con, 3)]
    assert "auffuellen" in gruende


def test_training_verwirft_ergebnis_nach_zusammenfuehren(con, monkeypatch):
    wahr, gids = _wolken(con, n_pro=30)
    for w in range(3):
        for lid in [l for l, x in wahr.items() if x == w][:10]:
            _bewerte(con, lid, gids[w])
    con.commit()
    echt = lernen._etiketten

    def mittendrin(c):  # während das Training rechnet, führt die App zwei Genres zusammen
        erg = echt(c)
        from musiksort import genres
        from pathlib import Path
        anderer_prozess = db.verbinden(Path(c.execute("PRAGMA database_list").fetchone()[2]))
        genres.zusammenfuehren(anderer_prozess, gids[2], gids[1])
        return erg
    monkeypatch.setattr(lernen, "_etiketten", mittendrin)
    assert lernen.trainieren(con) is False
    monkeypatch.setattr(lernen, "_etiketten", echt)
    assert lernen.trainieren(con) is True
    assert gids[2] not in {r[0] for r in con.execute("SELECT genre_id FROM vorhersagen")}


def test_mixtapes_werden_ueber_die_laenge_erkannt(con):
    """Gleicher Klang (House), aber Mixtapes dauern 30–90 min: nur die Länge trennt sie."""
    rng = np.random.default_rng(11)
    zentrum = rng.normal(size=1280)
    techno = rng.normal(size=1280)
    wahr = {}
    for art, anzahl in (("house", 150), ("mixtape", 150), ("techno", 150)):
        for _ in range(anzahl):
            v = (techno if art == "techno" else zentrum) + 0.6 * rng.normal(size=1280)
            v = (v / np.linalg.norm(v)).astype(np.float16)
            dauer = rng.uniform(1800, 5400) if art == "mixtape" else rng.uniform(180, 420)
            lid = con.execute("INSERT INTO lieder(pfad, status, emb, dauer) VALUES (?, 'analysiert', ?, ?)",
                              (f"{art}/{len(wahr)}.mp3", v.tobytes(), dauer)).lastrowid
            wahr[lid] = art
    gids = {a: con.execute("INSERT INTO genres(name) VALUES (?)", (a,)).lastrowid for a in ("house", "mixtape", "techno")}
    for art in gids:
        for lid in [l for l, a in wahr.items() if a == art][:50]:
            _bewerte(con, lid, gids[art])
    con.commit()
    assert lernen.trainieren(con) is True
    vorh = dict(con.execute("SELECT lied_id, genre_id FROM vorhersagen").fetchall())
    unbewertet = [l for l in wahr if l not in {r[0] for r in con.execute("SELECT lied_id FROM etiketten")}]
    richtig = sum(vorh[l] == gids[wahr[l]] for l in unbewertet) / len(unbewertet)
    assert richtig > 0.97   # ohne Länge wären House und Mixtape ein Münzwurf (~67 % gesamt)
    stand = {r[0]: r[1] for r in con.execute("SELECT genre_id, schwelle FROM modell_stand")}
    assert stand[gids["mixtape"]] is not None   # Mixtape darf selbst sortieren


def test_laenge_merkmale_robust():
    m = lernen.laenge_merkmale([240, 0, None, float("nan"), 900, 1500, 3600, 10**7])
    assert m.shape == (8, 4) and np.isfinite(m).all()
    assert (m[1] == m[0]).all() and (m[2] == m[0]).all()          # fehlend = wie 4 min
    assert m[4, 1] == 1 and m[4, 2] == 0 and m[6, 3] == 1          # > 10 min, nicht > 20; > 45 min


def test_obergenre_sortiert_grob_wenn_unter_genres_ununterscheidbar(con, monkeypatch):
    """A und B klingen gleich (eine Wolke, zufällig geteilt), C klar anders. A/B einzeln nie sicher,
    ihr Obergenre P aber schon: dann landen die Lieder grob in P – nie in C, nie fein in A oder B."""
    wahr, gids = _wolken(con, n_pro=150)
    g0, g1, g2 = gids
    p = con.execute("INSERT INTO genres(name) VALUES ('P')").lastrowid
    a = con.execute("INSERT INTO genres(name, eltern_id) VALUES ('A', ?)", (p,)).lastrowid
    b = con.execute("INSERT INTO genres(name, eltern_id) VALUES ('B', ?)", (p,)).lastrowid
    wolke = {w: [l for l, x in wahr.items() if x == w] for w in range(3)}
    for i, lid in enumerate(wolke[0][:80]):            # Wolke 0: halb A, halb B
        _bewerte(con, lid, a if i % 2 else b)
    for lid in wolke[1][:40]:
        _bewerte(con, lid, g1)
    for lid in wolke[2][:40]:
        _bewerte(con, lid, g2)
    con.commit()
    lernen.trainieren(con, cv_erzwingen=True)
    grob = dict(con.execute("SELECT lied_id, ober_id FROM vorhersagen WHERE ober_id IS NOT NULL").fetchall())
    assert len(grob) > 40 and set(grob.values()) == {p}
    assert all(wahr[l] == 0 for l in grob)                   # nur Lieder aus der A/B-Wolke
    assert con.execute("SELECT COUNT(*) FROM vorhersagen WHERE auto=1 AND genre_id IN (?,?)", (a, b)).fetchone()[0] == 0
    st = con.execute("SELECT ober_schwelle, ober_n FROM modell_stand WHERE genre_id=?", (p,)).fetchone()
    assert st[0] is not None and st[1] >= 32
    # Stichprobe im Obergenre: richtig, wenn deine Wahl darunter liegt
    con.execute("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)", (wolke[0][100], p, a))
    con.execute("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)", (wolke[0][101], p, g2))
    assert lernen._stichproben(con)[p] == (1, 2)
    # „Weiter“ wählt Stichproben auch aus dem Obergenre
    # naechste würfelt mit der Uhrzeit: für den Test jede Karte eine Stichprobe (sonst wackelt er)
    monkeypatch.setattr(config, "STICHPROBE_ANTEIL", 1.0)
    gezogen = [x for _ in range(40) for x in lernen.naechste(con, 3) if x["grund"] == "stichprobe"]
    assert any(x.get("pruef_id") == p for x in gezogen)
    # Vorrang: kurz vor dem Beleg kommen nur noch Stichproben – bis das Genre belegt ist
    monkeypatch.setattr(config, "STICHPROBE_ANTEIL", 0.0)
    monkeypatch.setattr(config, "STICHPROBE_VORRANG_LIEDER", 10)
    monkeypatch.setattr(config, "STICHPROBE_VORRANG_FEHLT", 40)
    for w, g in ((1, g1), (2, g2)):   # die fein sortierten Wolken sind schon belegt: kein Vorrang
        con.executemany("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)",
                        [(l, g, g) for l in wolke[w][100:115]])
    karten = lernen.naechste(con, 3)
    assert [x["grund"] for x in karten] == ["stichprobe"] * 3 and {x["pruef_id"] for x in karten} == {p}
    con.executemany("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)",
                    [(l, p, a) for l in wolke[0][102:140]])
    assert lernen.stichproben_fehlen(*lernen._stichproben(con)[p]) == 0
    assert all(x["grund"] != "stichprobe" for _ in range(5) for x in lernen.naechste(con, 3))


def test_deutschrap_wird_ueber_die_sprache_erkannt(con):
    """Gleicher Klang (Rap), aber deutsch vs. englisch gesungen: nur die gemessene Sprache trennt sie."""
    import json
    rng = np.random.default_rng(12)
    rap, rock = rng.normal(size=1280), rng.normal(size=1280)
    wahr = {}
    for art, anzahl in (("deutsch", 150), ("us", 150), ("rock", 150)):
        for _ in range(anzahl):
            v = (rock if art == "rock" else rap) + 0.6 * rng.normal(size=1280)
            v = (v / np.linalg.norm(v)).astype(np.float16)
            de = rng.uniform(0.8, 1.0) if art == "deutsch" else rng.uniform(0, 0.05)
            probs = json.dumps({"de": de, "en": 1 - de} if de > 0.5 else {"en": 1 - de, "de": de})
            lid = con.execute("INSERT INTO lieder(pfad, status, emb, dauer, sprache_probs) VALUES (?, 'analysiert', ?, 240, ?)",
                              (f"{art}/{len(wahr)}.mp3", v.tobytes(), probs)).lastrowid
            wahr[lid] = art
    gids = {a: con.execute("INSERT INTO genres(name) VALUES (?)", (a,)).lastrowid for a in ("deutsch", "us", "rock")}
    for art in gids:
        for lid in [l for l, a in wahr.items() if a == art][:50]:
            _bewerte(con, lid, gids[art])
    con.commit()
    assert lernen.trainieren(con) is True
    vorh = dict(con.execute("SELECT lied_id, genre_id FROM vorhersagen").fetchall())
    unbewertet = [l for l in wahr if l not in {r[0] for r in con.execute("SELECT lied_id FROM etiketten")}]
    assert sum(vorh[l] == gids[wahr[l]] for l in unbewertet) / len(unbewertet) > 0.97


def test_sprach_merkmale_robust():
    from musiksort import sprache
    m = sprache.merkmale([None, "", "kaputt", '{"de": 0.9, "nl": 0.06, "en": 0.04}'])
    assert m.shape == (4, sprache.BREITE)
    assert not m[:3].any()                       # nicht gemessen: alles 0, auch „gemessen“
    assert m[3, 0] == pytest.approx(0.9) and m[3, 1] == pytest.approx(0.04)
    assert m[3, len(sprache.SPRACHEN)] == pytest.approx(0.06) and m[3, -1] == 1   # Niederländisch unter „andere“
    assert sprache.anzeige("de", 0.95) == "Deutsch" and sprache.anzeige("en", 0.6) == "wohl Englisch"
    assert sprache.anzeige("de", 0.3) is None and sprache.anzeige("fehler", None) is None
    assert sprache.stellen(40, 10) == [10] and len(sprache.stellen(240, 60)) == 2


def test_karten_zeigen_nur_die_beste_fassung(con):
    wahr, gids = _wolken(con, n_pro=40)
    ids = list(wahr)
    # Familie 1: ids[0] (behalten) + ids[1]; Familie 2 über zwei Gruppen verkettet: ids[2] ~ ids[3] ~ ids[4]
    for g, mitglieder in enumerate(([(ids[0], 1), (ids[1], 0)], [(ids[2], 1), (ids[3], 0)], [(ids[3], 1), (ids[4], 0)])):
        gid = con.execute("INSERT INTO dup_gruppen(erstellt, status) VALUES (0, 'offen')").lastrowid
        con.executemany("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten) VALUES (?,?,?)",
                        [(gid, l, b) for l, b in mitglieder])
    _bewerte(con, ids[2], gids[0])   # Familie 2 ist damit erledigt
    con.commit()
    familie, nur_kopie = duplikate.kopien(con)
    assert familie[ids[2]] == familie[ids[3]] == familie[ids[4]] != familie[ids[0]]
    assert nur_kopie == {ids[1], ids[3], ids[4]}
    gesehen = [x["lied_id"] for _ in range(30) for x in lernen.naechste(con, 3)]
    assert not set(gesehen) & {ids[1], ids[3], ids[4]}       # nie Kopien
    erg = [x["lied_id"] for x in lernen.naechste(con, 10, ausschliessen=[ids[0]])]
    assert ids[0] not in erg and ids[1] not in erg


def test_kreuzvalidierung_trennt_kopien_nicht():
    """Zwei Kopien je Lied, Etiketten zufällig: ohne Familien „erkennt“ das Modell die Kopie wieder."""
    rng = np.random.default_rng(3)
    n = 150
    lieder = rng.normal(size=(n, 64))
    X = np.vstack([lieder, lieder + 0.01 * rng.normal(size=(n, 64))])
    y = np.tile(rng.integers(0, 3, n), 2)
    zaehl = {g: int((y == g).sum()) for g in set(y)}
    ohne = lernen._kreuzvalidierung(X, y, zaehl)
    mit = lernen._kreuzvalidierung(X, y, zaehl, familien=np.tile(np.arange(n), 2))
    assert (ohne[0] == ohne[1]).mean() > 0.6            # geschönt: Zufall wären ~33 %
    assert (mit[0] == mit[1]).mean() < 0.5              # ehrlich: nahe Zufall


def test_stichproben_dorthin_wo_sie_am_meisten_freischalten():
    # Zahlen wie am 29.10.: Deutsch HipHop 10/10 (2.690 warten), Hip Hop grob 27/28 (5.520),
    # Reggae belegt 22/22 (2.240), Rock grob 18/19 (480), Electronic 17/20 (wenige)
    assert [lernen.stichproben_fehlen(*s) for s in [(10, 10), (27, 28), (22, 22), (18, 19), (0, 0)]] == [5, 4, 0, 13, 15]
    w = dict(lernen.stichproben_gewichte({"dhh": 2690, "hh": 5520, "reggae": 2240, "rock": 480, "el": 30},
                                         {"dhh": (10, 10), "hh": (27, 28), "reggae": (22, 22), "rock": (18, 19),
                                          "el": (17, 20)}))
    anteil = {g: v / sum(w.values()) for g, v in w.items()}
    assert anteil["hh"] + anteil["dhh"] > 0.9      # fast alle Stichproben gehen dahin, wo 9 Antworten ~8.200 Lieder belegen
    assert 0 < anteil["reggae"] < 0.02             # belegt: nur noch selten nachgeprüft, aber nie ganz vergessen
    assert w["rock"] > w["el"] > 0


def test_widerlegtes_genre_wird_weiter_geprueft_und_kann_sich_erholen(con, monkeypatch):
    """Früher bekam ein widerlegtes Genre keine Stichproben mehr und blieb für immer aus (Electronic, 29.10.)."""
    wahr, gids = _wolken(con)
    je = {w: [l for l, x in wahr.items() if x == w] for w in range(3)}
    for w in range(3):
        for lid in je[w][:40]:
            _bewerte(con, lid, gids[w])
    # 3 von 12 Stichproben falsch → G0 widerlegt
    con.executemany("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)",
                    [(je[0][50 + i], gids[0], gids[1] if i < 3 else gids[0]) for i in range(12)])
    con.commit()
    lernen.trainieren(con)
    st = con.execute("SELECT * FROM modell_stand WHERE genre_id=?", (gids[0],)).fetchone()
    assert st["schwelle"] is None and st["widerlegt"] == 1 and st["pruef_schwelle"] is not None
    assert con.execute("SELECT COUNT(*) FROM vorhersagen WHERE auto=1 AND genre_id=?", (gids[0],)).fetchone()[0] == 0
    kandidaten = {r[0] for r in con.execute("SELECT lied_id FROM vorhersagen WHERE pruef_id=?", (gids[0],))}
    assert len(kandidaten) > 30 and all(wahr[l] == 0 for l in kandidaten)
    # „Weiter“ zieht weiter Stichproben für G0
    monkeypatch.setattr(config, "STICHPROBE_ANTEIL", 1.0)
    gezogen = [x for _ in range(30) for x in lernen.naechste(con, 3) if x["grund"] == "stichprobe"]
    assert any(x.get("pruef_id") == gids[0] for x in gezogen)
    # ein Training ohne Kreuzvalidierung behält die ruhende Schwelle
    lernen.trainieren(con)
    assert con.execute("SELECT pruef_schwelle FROM modell_stand WHERE genre_id=?", (gids[0],)).fetchone()[0] is not None
    # genug richtige Stichproben danach → wieder aktiv
    con.executemany("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,0)",
                    [(l, gids[0], gids[0]) for l in je[0][62:100]])
    con.commit()
    lernen.trainieren(con)
    st = con.execute("SELECT * FROM modell_stand WHERE genre_id=?", (gids[0],)).fetchone()
    assert st["schwelle"] is not None and st["widerlegt"] == 0 and st["pruef_schwelle"] is None
    assert con.execute("SELECT COUNT(*) FROM vorhersagen WHERE auto=1 AND genre_id=?", (gids[0],)).fetchone()[0] > 30
    assert con.execute("SELECT COUNT(*) FROM vorhersagen WHERE pruef_id IS NOT NULL").fetchone()[0] == 0
