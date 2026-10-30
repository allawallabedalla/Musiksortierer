"""Startmodell: eigene Bewertungen anonym weitergeben, damit eine neue Installation nicht bei null anfängt.

Ausgeführt beim Herausgeber (exportieren): je bewertetem Lied nur Klang-Steckbrief, Länge (volle Sekunden),
Gesangssprache und Genre – keine Pfade, Interpreten, Titel, IDs oder Zeitpunkte. Kopien desselben Lieds
zählen einmal, die Reihenfolge ist gemischt. Genres mit weniger als MIN_JE_GENRE Beispielen gehen im
nächsten Obergenre auf (oder fallen weg). Dazu ein daraus trainiertes Modell (Gewichte als Zahlen, kein Pickle).

Bei einer neuen Installation (laden): die Beispiele landen in der Tabelle startbeispiele, die Genres werden
angelegt. Das Training nimmt sie mit kleinerem Gewicht (config.START_GEWICHT) dazu, damit die Vorschläge
vom ersten Lied an etwas taugen. Trefferquoten, Schwellen und Freigaben zählen weiter nur eigene Bewertungen:
selbst sortiert wird erst, was die eigenen Bewertungen belegen.

  python -m musiksort.startmodell exportieren [ORDNER]   # Standard: startmodell/ im Projekt
  python -m musiksort.startmodell laden [ORDNER]
  python -m musiksort.startmodell entfernen
"""
import argparse
import json
import sys
import time

import numpy as np

from . import config, db, duplikate, lernen, sprache

ORDNER = config.PROJEKT / "startmodell"
MIN_JE_GENRE = 5
TRENNER = " › "
FORMAT = 1


def _pfade(con):
    namen = {g: (n, e) for g, n, e in con.execute("SELECT id, name, eltern_id FROM genres")}

    def pfad(g):
        teile, gesehen = [], set()
        while g is not None and g in namen and g not in gesehen:
            gesehen.add(g)
            teile.append(namen[g][0])
            g = namen[g][1]
        return TRENNER.join(reversed(teile))
    return {g: pfad(g) for g in namen}, {g: e for g, (_, e) in namen.items()}


def daten(con):
    """(E normiert float16 n×1280, dauer n, sprache n×BREITE ohne Gewicht, genre-Pfade n) – ohne Personenbezug."""
    pfade, eltern = _pfade(con)
    familie = duplikate.kopien(con)[0]
    zeilen = con.execute("SELECT e.lied_id, e.genre_id, l.emb, l.dauer, l.sprache_probs FROM etiketten e "
                         "JOIN lieder l ON l.id=e.lied_id WHERE e.haupt=1 AND l.status='analysiert' "
                         "AND l.emb IS NOT NULL ORDER BY e.ts").fetchall()
    gesehen, auswahl = set(), []
    for r in zeilen:  # Kopien desselben Lieds einmal
        f = familie.get(r["lied_id"], -r["lied_id"])
        if f not in gesehen:
            gesehen.add(f)
            auswahl.append(r)
    # zu kleine Genres im Obergenre aufgehen lassen (Unter-Genres gehören zum Obergenre)
    genre = [r["genre_id"] for r in auswahl]
    for _ in range(10):
        zaehl = {}
        for g in genre:
            zaehl[g] = zaehl.get(g, 0) + 1
        neu = [g if g is None or zaehl[g] >= MIN_JE_GENRE else eltern.get(g) for g in genre]
        if neu == genre:
            break
        genre = neu
    behalten = [i for i, g in enumerate(genre) if g is not None and zaehl.get(g, 0) >= MIN_JE_GENRE]
    rng = np.random.default_rng()
    rng.shuffle(behalten)  # Reihenfolge verrät nichts über den Zeitpunkt der Bewertung
    E = np.stack([np.frombuffer(auswahl[i]["emb"], np.float16) for i in behalten]).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    dauer = np.array([round(auswahl[i]["dauer"] or 0) for i in behalten], dtype=np.float32)
    spr = sprache.merkmale([auswahl[i]["sprache_probs"] for i in behalten]) / max(config.SPRACHE_GEWICHT, 1e-9)
    return E.astype(np.float16), dauer, np.round(spr, 3).astype(np.float16), [pfade[genre[i]] for i in behalten]


def exportieren(con=None, ordner=ORDNER):
    from sklearn.linear_model import LogisticRegression
    con = con or db.verbinden(schema=False)
    E, dauer, spr, genres = daten(con)
    liste = sorted(set(genres))
    y = np.array([liste.index(g) for g in genres])
    X = lernen._merkmale(E.astype(np.float32), dauer, spr.astype(np.float32) * config.SPRACHE_GEWICHT)
    modell = LogisticRegression(C=config.MODELL_C, max_iter=2000, class_weight="balanced").fit(X, y)
    ordner.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ordner / "steckbriefe.npz", klang=E, dauer=dauer, sprache=spr, genre=y.astype(np.int16))
    np.savez_compressed(ordner / "modell.npz", gewichte=modell.coef_.astype(np.float32),
                        achsen=modell.intercept_.astype(np.float32), klassen=modell.classes_.astype(np.int16))
    info = {"format": FORMAT, "stand": time.strftime("%Y-%m-%d"), "beispiele": len(genres), "genres": liste,
            "je_genre": {g: genres.count(g) for g in liste}, "merkmale": lernen.MERKMALE,
            "spalten": {"klang": 1280, "laenge": 4, "sprache": sprache.BREITE, "sprachen": list(sprache.SPRACHEN)},
            "gewichte": {"laenge": config.LAENGE_GEWICHT, "sprache": config.SPRACHE_GEWICHT},
            "modell": f"LogisticRegression(C={config.MODELL_C:g}, class_weight=balanced) auf [klang, lernen.laenge_merkmale(dauer), "
                      "sprache × Gewicht]; P = softmax(X @ gewichte.T + achsen)"}
    (ordner / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n")
    return info


def laden(con=None, ordner=ORDNER):
    """Beispiele übernehmen (ersetzt frühere Startbeispiele); fehlende Genres werden angelegt."""
    con = con or db.verbinden()
    info = json.loads((ordner / "info.json").read_text())
    if info.get("format") != FORMAT:
        raise ValueError(f"Startmodell-Format {info.get('format')} wird nicht unterstützt")
    d = np.load(ordner / "steckbriefe.npz", allow_pickle=False)
    with con:
        ids = [_genre_anlegen(con, pfad) for pfad in info["genres"]]
        con.execute("DELETE FROM startbeispiele")
        con.executemany("INSERT INTO startbeispiele(genre_id, emb, dauer, sprache) VALUES (?,?,?,?)",
                        [(ids[int(g)], k.astype(np.float16).tobytes(), float(t), s.astype(np.float16).tobytes())
                         for k, t, s, g in zip(d["klang"], d["dauer"], d["sprache"], d["genre"])])
        con.execute("INSERT OR REPLACE INTO modell_info VALUES ('startmodell', ?)", (info["stand"],))
    return len(d["genre"])


def _genre_anlegen(con, pfad):
    eltern = None
    for name in pfad.split(TRENNER):
        r = con.execute("SELECT id FROM genres WHERE name=? AND eltern_id IS ?", (name, eltern)).fetchone()
        eltern = r[0] if r else con.execute("INSERT INTO genres(name, eltern_id, erstellt) VALUES (?,?,?)",
                                             (name, eltern, db.jetzt())).lastrowid
    return eltern


def beispiele(con):
    """Für das Training: (E float32 normiert, dauer, sprache-Merkmale mit Gewicht, genre_ids) oder None."""
    zeilen = con.execute("SELECT s.genre_id, s.emb, s.dauer, s.sprache FROM startbeispiele s "
                         "JOIN genres g ON g.id=s.genre_id").fetchall()
    if not zeilen:
        return None
    E = np.stack([np.frombuffer(r[1], np.float16) for r in zeilen]).astype(np.float32)
    spr = np.stack([np.frombuffer(r[3], np.float16) for r in zeilen]).astype(np.float32) * config.SPRACHE_GEWICHT
    return E, np.array([r[2] for r in zeilen], np.float32), spr, np.array([r[0] for r in zeilen])


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("aktion", choices=["exportieren", "laden", "entfernen"])
    a.add_argument("ordner", nargs="?", default=str(ORDNER))
    args = a.parse_args()
    from pathlib import Path
    if args.aktion == "exportieren":
        i = exportieren(ordner=Path(args.ordner))
        print(f"{i['beispiele']} Beispiele in {len(i['genres'])} Genres nach {args.ordner}")
    elif args.aktion == "laden":
        print(f"{laden(ordner=Path(args.ordner))} Startbeispiele übernommen; wirken beim nächsten Training")
    else:
        c = db.verbinden()
        with c:
            n = c.execute("DELETE FROM startbeispiele").rowcount
            c.execute("DELETE FROM modell_info WHERE schluessel='startmodell'")
        print(f"{n} Startbeispiele entfernt")
    sys.exit(0)
