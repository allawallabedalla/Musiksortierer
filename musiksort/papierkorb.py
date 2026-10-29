"""Papierkorb für den Sortierlauf: nichts wird sofort gelöscht.

Vorgemerkte Duplikate wandern nach  <Media>/_papierkorb/<Datum>/<alter Pfad>  (gleiche Freigabe, also nur
umbenennen, kein Kopieren). Erst nach PAPIERKORB_TAGE werden sie endgültig gelöscht – Datei für Datei,
nie ein ganzer Ordnerbaum. Bis dahin holt `zurueckholen` jede Datei an ihren alten Platz.

Jede Bewegung steht im Protokoll (aktion, von, nach, lied_id). Das Lied bekommt den Status „papierkorb“
(die nächtliche Suche lässt es in Ruhe, der Ordner _papierkorb ist von der Suche ausgeschlossen),
nach dem endgültigen Löschen „geloescht“. Bewertungen bleiben in der DB.

Wird vom Sortierlauf benutzt (noch nicht gebaut); die Medien-Freigabe muss dafür beschreibbar eingebunden sein.
"""
import datetime
import json
import os
from pathlib import Path

from . import config, db


class PapierkorbFehler(RuntimeError):
    pass


def ordner():
    return config.MEDIA_ROOT / config.PAPIERKORB


def _protokoll(con, aktion, lied_id, von, nach, info=None):
    con.execute("INSERT INTO protokoll(ts, aktion, von, nach, lied_id) VALUES (?,?,?,?,?)",
                (db.jetzt(), aktion, von, json.dumps({"nach": nach, **(info or {})}, ensure_ascii=False), lied_id))


def _freier_name(ziel: Path) -> Path:
    if not ziel.exists():
        return ziel
    for i in range(2, 1000):
        k = ziel.with_name(f"{ziel.stem} ({i}){ziel.suffix}")
        if not k.exists():
            return k
    raise PapierkorbFehler(f"kein freier Name für {ziel}")


def verschieben(con, lied_id, grund, heute=None):
    """Eine Datei in den Papierkorb (Aufrufer hält die Transaktion). Rückgabe: neuer relativer Pfad."""
    r = con.execute("SELECT pfad, status FROM lieder WHERE id=?", (lied_id,)).fetchone()
    if not r:
        raise PapierkorbFehler("Lied unbekannt")
    if r[1] == "papierkorb":
        raise PapierkorbFehler("liegt schon im Papierkorb")
    quelle = config.medienpfad(r[0])
    if not quelle.is_file():
        raise PapierkorbFehler(f"Datei fehlt: {r[0]}")
    tag = (heute or datetime.date.today()).isoformat()
    ziel = _freier_name(config.medienpfad(f"{config.PAPIERKORB}/{tag}/{r[0]}"))
    ziel.parent.mkdir(parents=True, exist_ok=True)
    os.rename(quelle, ziel)  # gleiche Freigabe: nur umbenennen
    neu = str(ziel.relative_to(config.MEDIA_ROOT.resolve()))
    con.execute("UPDATE lieder SET status='papierkorb', pfad=? WHERE id=?", (neu, lied_id))
    _protokoll(con, "papierkorb", lied_id, r[0], neu, {"grund": grund})
    return neu


def gruppe_ausfuehren(con, gruppe_id):
    """Freigegebene Duplikatgruppe ausführen: alle nicht behaltenen Dateien in den Papierkorb, Gruppe „erledigt“."""
    g = con.execute("SELECT status FROM dup_gruppen WHERE id=?", (gruppe_id,)).fetchone()
    if not g or g[0] != "freigegeben":
        raise PapierkorbFehler("Gruppe ist nicht freigegeben")
    weg = [r[0] for r in con.execute("SELECT lied_id FROM dup_mitglieder WHERE gruppe_id=? AND behalten=0", (gruppe_id,))]
    # Favorit geht nicht verloren: hatte eine aussortierte Datei den Stern, bekommt ihn die behaltene
    if weg and con.execute(f"SELECT 1 FROM lieder WHERE stern=1 AND id IN ({','.join('?' * len(weg))})", weg).fetchone():
        con.execute("UPDATE lieder SET stern=1, stern_ts=? WHERE stern=0 AND id IN "
                    "(SELECT lied_id FROM dup_mitglieder WHERE gruppe_id=? AND behalten=1)", (db.jetzt(), gruppe_id))
    # Bewertung geht nicht verloren: ist die behaltene Datei unbewertet, übernimmt sie die jüngste Bewertung
    # einer aussortierten (Genre + Zusätze); eigene Bewertungen der behaltenen bleiben unangetastet
    behalten = [r[0] for r in con.execute("SELECT lied_id FROM dup_mitglieder WHERE gruppe_id=? AND behalten=1",
                                          (gruppe_id,))]
    if weg and behalten and not con.execute("SELECT 1 FROM etiketten WHERE lied_id=?", (behalten[0],)).fetchone():
        quelle = con.execute(f"SELECT lied_id FROM etiketten WHERE haupt=1 AND lied_id IN ({','.join('?' * len(weg))}) "
                             "ORDER BY ts DESC LIMIT 1", weg).fetchone()
        if quelle:
            con.execute("INSERT INTO etiketten SELECT ?, genre_id, haupt, quelle, ts FROM etiketten WHERE lied_id=?",
                        (behalten[0], quelle[0]))
    for lid in weg:
        verschieben(con, lid, f"Duplikat (Gruppe {gruppe_id})")
    con.execute("UPDATE dup_gruppen SET status='erledigt' WHERE id=?", (gruppe_id,))
    return len(weg)


def zurueckholen(con, lied_id):
    """Datei aus dem Papierkorb an ihren alten Platz (Aufrufer hält die Transaktion)."""
    r = con.execute("SELECT pfad, status FROM lieder WHERE id=?", (lied_id,)).fetchone()
    if not r or r[1] != "papierkorb":
        raise PapierkorbFehler("liegt nicht im Papierkorb")
    alt = con.execute("SELECT von FROM protokoll WHERE lied_id=? AND aktion='papierkorb' ORDER BY id DESC LIMIT 1",
                      (lied_id,)).fetchone()
    if not alt:
        raise PapierkorbFehler("alter Pfad unbekannt")
    quelle, ziel = config.medienpfad(r[0]), config.medienpfad(alt[0])
    if ziel.exists():
        raise PapierkorbFehler(f"am alten Platz liegt schon eine Datei: {alt[0]}")
    ziel.parent.mkdir(parents=True, exist_ok=True)
    os.rename(quelle, ziel)
    con.execute("UPDATE lieder SET status='analysiert', pfad=? WHERE id=?", (alt[0], lied_id))
    # die Gruppe gilt nicht mehr als ausgeführt: bei der nächsten Duplikatsuche neu vorlegen
    con.execute("UPDATE dup_gruppen SET status='ersetzt' WHERE status='erledigt' AND id IN "
                "(SELECT gruppe_id FROM dup_mitglieder WHERE lied_id=?)", (lied_id,))
    _protokoll(con, "papierkorb_zurueck", lied_id, r[0], alt[0])


def leeren(con, tage=None, heute=None):
    """Tagesordner älter als `tage` endgültig löschen – Datei für Datei. Rückgabe: Anzahl gelöschter Dateien."""
    tage = config.PAPIERKORB_TAGE if tage is None else tage
    grenze = (heute or datetime.date.today()) - datetime.timedelta(days=tage)
    wurzel = ordner()
    if not wurzel.is_dir():
        return 0
    n = 0
    for tagesordner in sorted(wurzel.iterdir()):
        try:
            tag = datetime.date.fromisoformat(tagesordner.name)
        except ValueError:
            continue  # nur eigene Tagesordner anfassen
        if tag >= grenze:
            continue
        for f in sorted(tagesordner.rglob("*"), key=lambda p: -len(p.parts)):
            if f.is_file() and not f.is_symlink():
                rel = str(f.resolve().relative_to(config.MEDIA_ROOT.resolve()))
                f.unlink()
                n += 1
                with_id = con.execute("SELECT id FROM lieder WHERE pfad=? AND status='papierkorb'", (rel,)).fetchone()
                if with_id:
                    con.execute("UPDATE lieder SET status='geloescht' WHERE id=?", (with_id[0],))
                    _protokoll(con, "geloescht", with_id[0], rel, None, {"nach_tagen": tage})
            elif f.is_dir() and not any(f.iterdir()):
                f.rmdir()
        if not any(tagesordner.iterdir()):
            tagesordner.rmdir()
    return n
