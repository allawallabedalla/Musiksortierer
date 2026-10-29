"""Tägliche, konsistente Sicherung der Datenbank (Bewertungen = deine eigentliche Arbeit).

Nutzt die SQLite-Backup-Schnittstelle (auch während App und Analyse schreiben),
behält die letzten 7 Tage. Die Kopien liegen neben der DB; eine Sicherung des Rechners
erfasst sie dann als ruhende Dateien.

  python -m musiksort.sicherung
"""
import datetime
import sqlite3

from . import config

BEHALTEN = 7


def sichern(ziel_ordner=None, behalten=BEHALTEN):
    ziel_ordner = ziel_ordner or config.DATEN / "sicherung"
    ziel_ordner.mkdir(parents=True, exist_ok=True)
    ziel = ziel_ordner / f"musik-{datetime.date.today().isoformat()}.db"
    tmp = ziel.with_suffix(".tmp")
    quelle = sqlite3.connect(f"file:{config.DB_PFAD}?mode=ro", uri=True)
    kopie = sqlite3.connect(tmp)
    with kopie:
        quelle.backup(kopie)
    pruefung = kopie.execute("PRAGMA integrity_check").fetchone()[0]
    kopie.close()
    quelle.close()
    if pruefung != "ok":
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Sicherung nicht konsistent: {pruefung}")
    tmp.replace(ziel)
    for alt in sorted(ziel_ordner.glob("musik-*.db"))[:-behalten]:
        alt.unlink()
    return ziel


# Deine eigentliche Arbeit, klein genug für eine stündliche Datensicherung:
# Genres, Bewertungen, Stichproben, „Später“, eigene Duplikat-Entscheidungen. Alles andere lässt sich neu rechnen.
EIGENE_ARBEIT = {
    "genres": "SELECT * FROM genres",
    "etiketten": "SELECT * FROM etiketten",
    "stichproben": "SELECT * FROM stichproben",
    "uebersprungen": "SELECT * FROM uebersprungen",
    "dup_gruppen": "SELECT * FROM dup_gruppen WHERE entschieden_von='ich'",
    "dup_mitglieder": "SELECT m.* FROM dup_mitglieder m JOIN dup_gruppen g ON g.id=m.gruppe_id WHERE g.entschieden_von='ich'",
    "lieder": "SELECT id, pfad, stern, stern_ts FROM lieder WHERE stern=1 OR id IN (SELECT lied_id FROM etiketten UNION "
              "SELECT m.lied_id FROM dup_mitglieder m JOIN dup_gruppen g ON g.id=m.gruppe_id WHERE g.entschieden_von='ich')",
}


def bewertungen_exportieren(ziel=None):
    """Eigene Arbeit als JSON (atomar ersetzt). Rückgabe: (Pfad, Anzahl Bewertungen)."""
    import json
    import os
    ziel = ziel or config.DATEN / "sicherung" / "bewertungen.json"
    ziel.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f"file:{config.DB_PFAD}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    daten = {"stand": datetime.datetime.now().isoformat(timespec="seconds")}
    with con:  # eine Momentaufnahme
        con.execute("BEGIN")
        for name, sql in EIGENE_ARBEIT.items():
            daten[name] = [dict(r) for r in con.execute(sql)]
    con.close()
    tmp = ziel.with_suffix(".tmp")
    tmp.write_text(json.dumps(daten, ensure_ascii=False, default=lambda b: None))
    os.replace(tmp, ziel)
    return ziel, len({e["lied_id"] for e in daten["etiketten"]})


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--bewertungen"]:
        pfad, n = bewertungen_exportieren()
        print(f"{n} Bewertungen nach {pfad}")
    else:
        print("gesichert:", sichern())
