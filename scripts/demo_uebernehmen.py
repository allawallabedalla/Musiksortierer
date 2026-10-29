"""Bewertungen und Genres aus einer Demo-DB in die echte DB übernehmen (einmalig).

  .venv/bin/python scripts/demo_uebernehmen.py ~/.cache/musiksortierer/demo/musik.db

Lieder werden über den Pfad zugeordnet. Analyse-Ergebnisse der Demo werden mit
übernommen, damit die bewerteten Lieder nicht erneut analysiert werden müssen.
Bestehende Etiketten in der Ziel-DB bleiben; doppelte werden übersprungen.
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from musiksort import db  # noqa: E402

ANALYSE = ["status", "fehler", "codec", "bitrate", "samplerate", "bittiefe", "kanaele", "dauer",
           "verlustfrei", "interpret", "titel", "album", "tag_genre", "bpm", "snippet_start",
           "snippet_quelle", "emb", "discogs", "fp", "analysiert", "analyse_s"]


def uebernehmen(quelle_pfad, ziel):
    q = sqlite3.connect(f"file:{quelle_pfad}?mode=ro", uri=True)
    q.row_factory = sqlite3.Row
    # Genres in Baum-Reihenfolge (Eltern vor Kindern), über Name+Eltern zuordnen
    alt_zu_neu, offen = {}, q.execute("SELECT id, name, eltern_id FROM genres").fetchall()
    while offen:
        rest = []
        for g in offen:
            if g["eltern_id"] is not None and g["eltern_id"] not in alt_zu_neu:
                rest.append(g)
                continue
            eltern = alt_zu_neu.get(g["eltern_id"])
            z = ziel.execute("SELECT id FROM genres WHERE name=? AND eltern_id IS ?", (g["name"], eltern)).fetchone()
            alt_zu_neu[g["id"]] = z[0] if z else ziel.execute(
                "INSERT INTO genres(name, eltern_id, erstellt) VALUES (?,?,?)", (g["name"], eltern, db.jetzt())).lastrowid
        if len(rest) == len(offen):
            raise SystemExit("Genre-Baum der Demo ist kaputt (Kreis?)")
        offen = rest

    lieder = analysen = etiketten = fehlend = 0
    for l in q.execute("SELECT * FROM lieder"):
        z = ziel.execute("SELECT id, status FROM lieder WHERE pfad=?", (l["pfad"],)).fetchone()
        if z is None:
            fehlend += 1
            continue
        lieder += 1
        if l["status"] == "analysiert" and z["status"] != "analysiert":
            ziel.execute(f"UPDATE lieder SET {', '.join(s + '=?' for s in ANALYSE)} WHERE id=?",
                         [l[s] for s in ANALYSE] + [z["id"]])
            analysen += 1
        if ziel.execute("SELECT 1 FROM etiketten WHERE lied_id=? AND haupt=1", (z["id"],)).fetchone():
            continue  # schon bewertet: nie ein zweites Hauptgenre daneben setzen
        for e in q.execute("SELECT * FROM etiketten WHERE lied_id=?", (l["id"],)):
            r = ziel.execute("INSERT OR IGNORE INTO etiketten VALUES (?,?,?,?,?)",
                             (z["id"], alt_zu_neu[e["genre_id"]], e["haupt"], e["quelle"], e["ts"]))
            etiketten += r.rowcount
    ziel.commit()
    return {"genres": len(alt_zu_neu), "lieder": lieder, "analysen": analysen,
            "etiketten": etiketten, "nicht_gefunden": fehlend}


if __name__ == "__main__":
    print(uebernehmen(Path(sys.argv[1]).expanduser(), db.verbinden()))
