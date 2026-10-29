"""Duplikate über den Chromaprint-Fingerabdruck finden.

Gleiche Aufnahme = Fingerabdrücke stimmen bei bestem Versatz bis auf
wenige Bits überein UND die Dauer ist fast gleich. Jedes Mitglied einer
„sicheren“ Gruppe erfüllt das direkt gegenüber der behaltenen Datei (keine
Ketten a~b~c~d). Ähnliche Fassungen (Edit, Remaster, Live) werden als
Zweifelsfall gemeldet und nie automatisch gruppiert.

Gruppen-IDs werden nie wiederverwendet: alte offene Gruppen werden als
„ersetzt“ markiert statt gelöscht. Eine Freigabe in der App bezieht sich
damit immer auf genau die Gruppe, die angezeigt wurde.

  python -m musiksort.duplikate
"""
import json

import numpy as np

from . import config, db, klang

MAX_BITFEHLER = 0.15     # gleiche Aufnahme liegt typisch < 0,1; fremde ~0,5
MAX_VERSATZ = 12         # ~1,5 s Versatz am Anfang (Stille, Encoder-Delay)
DAUER_TOLERANZ = 0.01    # 1 % bzw. mindestens 2,5 s: lieber einmal mehr fragen als eine Fassung löschen
ZWEIFEL_BITFEHLER = 0.25  # ähnlich, aber nicht sicher gleich
ZWEIFEL_DAUER = 20.0     # s; Radio-Edit vs. Original fällt meist darunter
MIN_GLEICHE_WERTE = 6    # Kandidat, wenn so viele Teilwerte exakt übereinstimmen
MAX_BLOCK = 50           # sehr häufige Werte (Stille) tragen nichts bei
INDEX_POSITIONEN = slice(40, 200)  # Mittelteil des Abdrucks für den Index

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# offene Gruppen, deren Frage noch nicht durch eine andere Freigabe erledigt ist
OFFENE_GRUPPEN = ("g.status='offen' AND NOT EXISTS (SELECT 1 FROM dup_mitglieder m2 JOIN dup_mitglieder x "
                  "ON x.lied_id=m2.lied_id JOIN dup_gruppen g2 ON g2.id=x.gruppe_id WHERE m2.gruppe_id=g.id "
                  "AND x.behalten=0 AND g2.status='freigegeben' AND g2.id!=g.id)")


def bitfehler(a, b):
    beste = 1.0
    for v in range(-MAX_VERSATZ, MAX_VERSATZ + 1):
        x, y = (a[v:], b) if v >= 0 else (a, b[-v:])
        n = min(len(x), len(y))
        if n < 50:
            continue
        diff = np.bitwise_xor(x[:n], y[:n]).view(np.uint8)
        beste = min(beste, _POP[diff].sum() / (n * 32))
    return beste


def qualitaet(r):
    """Höher ist besser: verlustfrei vor Bitrate, dann Bittiefe/Abtastrate, dann Tags,
    dann der Ordner (bei Gleichstand lieber aus einer gepflegten Sammlung als aus einem Duplikat-Ordner)."""
    tags = sum(bool(r[k]) for k in ("interpret", "titel", "album"))
    ordner = config.ORDNER_RANG.get(r["pfad"].split("/", 1)[0], 0)
    if klang.falsche_verlustfrei(r):  # aufgeblasene MP3 in FLAC/WAV: zählt wie die MP3, die sie war
        return (0, klang.geschaetzte_bitrate(r["grenzfrequenz"]), 0, 0, tags, ordner, -len(r["pfad"]))
    return (r["verlustfrei"] or 0, r["bitrate"] or 0 if not r["verlustfrei"] else 0,
            r["bittiefe"] or 0, r["samplerate"] or 0, tags, ordner, -len(r["pfad"]))


def dauer_gleich(da, db_):
    return abs(da - db_) <= max(2.5, DAUER_TOLERANZ * max(da, db_))


def kandidaten(fps):
    """Paare mit ≥ MIN_GLEICHE_WERTE exakt gleichen Index-Werten.

    Speicherschonend: ein sortiertes Werte-Array, Blöcke über Grenz-Indizes
    (kein Array-Objekt pro Wert).
    """
    werte, lieder = [], []
    for lid, fp in fps.items():
        teil = np.unique(fp[INDEX_POSITIONEN])
        werte.append(teil)
        lieder.append(np.full(teil.size, lid, dtype=np.int64))
    if not werte:
        return {}
    werte, lieder = np.concatenate(werte), np.concatenate(lieder)
    ordnung = np.argsort(werte, kind="stable")
    werte, lieder = werte[ordnung], lieder[ordnung]
    del ordnung
    anfaenge = np.flatnonzero(np.r_[True, werte[1:] != werte[:-1]])
    groessen = np.diff(np.r_[anfaenge, werte.size])
    paare = {}
    for a, g in zip(anfaenge[(groessen > 1) & (groessen <= MAX_BLOCK)],
                    groessen[(groessen > 1) & (groessen <= MAX_BLOCK)]):
        b = np.unique(lieder[a:a + g])
        for i in range(b.size):
            for j in range(i + 1, b.size):
                k = (int(b[i]), int(b[j]))
                paare[k] = paare.get(k, 0) + 1
    return paare


def finden(con=None):
    con = con or db.verbinden()
    from .scan import gruppen_ungueltig_machen
    with con:  # Entscheidungen über inzwischen geänderte/fehlende Dateien gelten nicht mehr
        gruppen_ungueltig_machen(con)
    zeilen = con.execute("SELECT id, pfad, fp, dauer, verlustfrei, bitrate, bittiefe, samplerate, interpret, titel, album, "
                         "grenzfrequenz, kante "
                         "FROM lieder WHERE status='analysiert' AND fp IS NOT NULL").fetchall()
    fps = {r["id"]: np.frombuffer(r["fp"], np.uint32) for r in zeilen}
    info = {r["id"]: r for r in zeilen}
    if not fps:
        print("keine Fingerabdrücke")
        return

    # Entschiedenes bleibt: Lieder, die nach Freigabe gelöscht werden, tauchen nicht
    # wieder auf; eine „alle behalten“-Gruppe wird nicht erneut gemeldet.
    entschieden = {}
    for r in con.execute("SELECT g.id, g.status, m.lied_id, m.behalten FROM dup_gruppen g "
                         "JOIN dup_mitglieder m ON m.gruppe_id=g.id WHERE g.status IN ('freigegeben','verworfen','erledigt')"):
        entschieden.setdefault(r["id"], []).append((r["lied_id"], r["status"], r["behalten"]))
    wird_geloescht = {lid for m in entschieden.values() for lid, st, beh in m if st == "freigegeben" and not beh}
    schon_gesehen = [frozenset(lid for lid, _, _ in m) for m in entschieden.values()]

    eltern = {}

    def wurzel(x):
        while eltern.get(x, x) != x:
            eltern[x] = eltern.get(eltern[x], eltern[x])
            x = eltern[x]
        return x

    geprueft = 0
    zweifel = {}
    for (a, b), n in kandidaten(fps).items():
        if n < MIN_GLEICHE_WERTE or a in wird_geloescht or b in wird_geloescht:
            continue
        da, dbb = info[a]["dauer"] or 0, info[b]["dauer"] or 0
        if abs(da - dbb) > ZWEIFEL_DAUER:
            continue
        geprueft += 1
        bf = bitfehler(fps[a], fps[b])
        if bf <= MAX_BITFEHLER and dauer_gleich(da, dbb):
            eltern[wurzel(a)] = wurzel(b)
        elif bf <= ZWEIFEL_BITFEHLER:
            zweifel[(a, b)] = bf  # gleicher Anfang, aber andere Länge/Fassung: dich fragen

    gruppen = {}
    for x in set(eltern) | set(eltern.values()):
        gruppen.setdefault(wurzel(x), set()).add(x)

    # Jedes Mitglied direkt gegen die behaltene Datei prüfen (keine Ketten)
    sichere = []
    for mitglieder in gruppen.values():
        best = max(mitglieder, key=lambda m: qualitaet(info[m]))
        gut = {best}
        for m in mitglieder - {best}:
            bf = bitfehler(fps[best], fps[m])
            if bf <= MAX_BITFEHLER and dauer_gleich(info[best]["dauer"] or 0, info[m]["dauer"] or 0):
                gut.add(m)
            elif bf <= ZWEIFEL_BITFEHLER:
                zweifel[(min(best, m), max(best, m))] = bf
        if len(gut) >= 2 and not any(gut <= s for s in schon_gesehen):
            sichere.append(gut)

    # Zweifel nur einmal je Gruppenpaar, jeweils mit dem besten Vertreter
    vertreter_von = {m: max(g, key=lambda x: qualitaet(info[x])) for g in sichere for m in g}
    zweifel_paare = set()
    for a, b in zweifel:
        va, vb = vertreter_von.get(a, a), vertreter_von.get(b, b)
        if va != vb and not any({va, vb} <= s for s in schon_gesehen):
            zweifel_paare.add((min(va, vb), max(va, vb)))

    with con:
        # alte offene Gruppen nicht löschen, sondern ersetzen: IDs bleiben eindeutig
        con.execute("DELETE FROM dup_mitglieder WHERE gruppe_id IN (SELECT id FROM dup_gruppen WHERE status='offen')")
        con.execute("UPDATE dup_gruppen SET status='ersetzt' WHERE status='offen'")
        for g in sichere:
            _speichern(con, g, 1, info, fps)
        for a, b in sorted(zweifel_paare):
            _speichern(con, {a, b}, 0, info, fps)
    ueberzaehlig = sum(len(g) - 1 for g in sichere)
    print(f"{len(fps)} Lieder, {geprueft} Kandidatenpaare geprüft, {len(sichere)} Gruppen, "
          f"{ueberzaehlig} überzählige Dateien, {len(zweifel_paare)} Zweifelsfälle")


def _speichern(con, mitglieder, sicher, info, fps):
    # int(): numpy-Zahlen landen in SQLite sonst als BLOB
    sortiert = sorted((int(m) for m in mitglieder), key=lambda m: qualitaet(info[m]), reverse=True)
    best = sortiert[0]
    gid = con.execute("INSERT INTO dup_gruppen(erstellt, sicher) VALUES (?,?)", (db.jetzt(), sicher)).lastrowid
    for m in sortiert:
        con.execute("INSERT INTO dup_mitglieder(gruppe_id, lied_id, behalten, abweichung) VALUES (?,?,?,?)",
                    (gid, m, int(m == best), 0.0 if m == best else round(bitfehler(fps[best], fps[m]), 3)))


class Abgelehnt(Exception):
    """Entscheidung nicht zulässig (code wie HTTP: 400/404/409)."""

    def __init__(self, code, text):
        super().__init__(text)
        self.code, self.text = code, text


def kopien(con):
    """Wer ist eine Kopie von wem? Über alle offenen und freigegebenen Gruppen (auch Zweifelsfälle).

    Rückgabe (familie, nur_kopie): familie = {lied_id: Kennung der Kopienfamilie} (Ketten über mehrere
    Gruppen zusammengefasst), nur_kopie = Lieder, die in irgendeiner Gruppe nicht die behaltene Datei sind.
    Für die Bewertung: nur die beste Fassung zeigen; für die Kreuzvalidierung: Familien nie aufteilen.
    """
    eltern = {}

    def wurzel(x):
        while eltern.get(x, x) != x:
            eltern[x] = eltern.get(eltern[x], eltern[x])
            x = eltern[x]
        return x

    nur_kopie, erstes = set(), {}
    for gid, lid, behalten in con.execute(
            "SELECT m.gruppe_id, m.lied_id, m.behalten FROM dup_mitglieder m JOIN dup_gruppen g ON g.id=m.gruppe_id "
            "WHERE g.status IN ('offen','freigegeben')"):
        if not behalten:
            nur_kopie.add(lid)
        if gid in erstes:
            a, b = wurzel(erstes[gid]), wurzel(lid)
            if a != b:
                eltern[b] = a
        else:
            erstes[gid] = lid
            eltern.setdefault(lid, lid)
    return {lid: wurzel(lid) for lid in list(eltern)}, nur_kopie


def protokoll(con, aktion, info, wer=None):
    """Jede folgenreiche Entscheidung nachvollziehbar (wer, wann, was)."""
    con.execute("INSERT INTO protokoll(ts, aktion, von, nach) VALUES (?,?,?,?)",
                (db.jetzt(), aktion, wer, json.dumps(info, ensure_ascii=False)))


def entscheiden(con, gruppe_id, status, mitglieder, behalten=None, von="ich", wer=None, grund=None):
    """Eine offene Gruppe entscheiden: freigegeben (behalten bleibt, Rest wird vorgemerkt) oder verworfen.

    Gemeinsam für deine Klicks und die automatischen Stufen; der Aufrufer hält die Transaktion
    (BEGIN IMMEDIATE … commit). Abgelehnt wird alles, was zusammen mit anderen Entscheidungen
    eine Fassung ganz verlieren könnte.
    """
    if status not in ("freigegeben", "verworfen"):
        raise Abgelehnt(400, "Status ungültig")
    if not con.execute("SELECT 1 FROM dup_gruppen WHERE id=? AND status='offen'", (gruppe_id,)).fetchone():
        raise Abgelehnt(404, "Gruppe ist schon entschieden oder neu berechnet. Bitte neu laden.")
    drin = con.execute("SELECT m.lied_id, m.behalten, l.status, l.pfad FROM dup_mitglieder m "
                       "JOIN lieder l ON l.id=m.lied_id WHERE m.gruppe_id=?", (gruppe_id,)).fetchall()
    if sorted(m[0] for m in drin) != sorted(mitglieder):
        raise Abgelehnt(409, "Gruppe hat sich geändert. Bitte neu laden.")
    if any(m[2] != "analysiert" for m in drin):
        raise Abgelehnt(409, "Eine der Dateien hat sich geändert oder fehlt. Die Gruppe wird neu geprüft.")
    for m in drin:  # schon anderswo zum Löschen vorgemerkt: diese Frage ist erledigt
        if con.execute("SELECT 1 FROM dup_mitglieder x JOIN dup_gruppen g ON g.id=x.gruppe_id WHERE x.lied_id=? "
                       "AND x.behalten=0 AND g.status='freigegeben' AND g.id!=?", (m[0], gruppe_id)).fetchone():
            raise Abgelehnt(409, f"„{m[3].rsplit('/', 1)[-1]}“ ist schon in einer anderen Gruppe zum Löschen "
                                 "vorgemerkt. Bitte neu laden.")
    behalten = behalten if behalten is not None else next(m[0] for m in drin if m[1])
    if behalten not in {m[0] for m in drin}:
        raise Abgelehnt(400, "nicht in der Gruppe")
    if status == "freigegeben":
        for m in drin:
            if m[0] == behalten:
                continue
            if con.execute("SELECT 1 FROM dup_mitglieder x JOIN dup_gruppen g ON g.id=x.gruppe_id WHERE x.lied_id=? "
                           "AND x.behalten=1 AND g.id!=? AND g.status IN ('offen','freigegeben')",
                           (m[0], gruppe_id)).fetchone():
                raise Abgelehnt(409, f"„{m[3].rsplit('/', 1)[-1]}“ ist in einer anderen Gruppe die behaltene Datei. "
                                     "Dort zuerst entscheiden oder hier alle behalten.")
        if con.execute("SELECT 1 FROM dup_mitglieder x JOIN dup_gruppen g ON g.id=x.gruppe_id WHERE x.lied_id=? "
                       "AND x.behalten=0 AND g.status='freigegeben'", (behalten,)).fetchone():
            raise Abgelehnt(409, "Die behaltene Datei ist in einer anderen Gruppe schon zum Löschen vorgemerkt.")
    con.execute("UPDATE dup_mitglieder SET behalten=(lied_id=?) WHERE gruppe_id=?", (behalten, gruppe_id))
    con.execute("UPDATE dup_gruppen SET status=?, entschieden_von=? WHERE id=?", (status, von, gruppe_id))
    protokoll(con, ("duplikat_" if von == "ich" else "duplikat_auto_") + status,
              {"gruppe": gruppe_id, "mitglieder": list(mitglieder), "behalten": behalten, "von": von,
               **({"grund": grund} if grund else {})}, wer)
    return behalten


def zuruecknehmen(con, gruppe_id, wer=None):
    """Entscheidung zurücknehmen (nur solange noch nichts gelöscht wurde).

    Steckt ein Mitglied inzwischen in einer anderen offenen Gruppe oder hat sich eine Datei
    geändert, kommt die Gruppe nicht zurück, sondern wird bei der nächsten Duplikatsuche neu
    vorgelegt (keine doppelte Vorlage derselben Frage). Rückgabe: neuer Status.
    Überschneidungen offener Gruppen gibt es sonst durchaus (Zweifelspaare teilen sich den
    Vertreter); sicher ist das, weil jede Freigabe gegen alle anderen Gruppen geprüft wird.
    """
    r = con.execute("SELECT status, entschieden_von FROM dup_gruppen WHERE id=?", (gruppe_id,)).fetchone()
    if not r or r[0] not in ("freigegeben", "verworfen"):
        raise Abgelehnt(404, "Nichts zurückzunehmen")
    ueberschneidung = con.execute(
        "SELECT 1 FROM dup_mitglieder m JOIN dup_gruppen g ON g.id=m.gruppe_id WHERE g.status='offen' "
        "AND m.lied_id IN (SELECT lied_id FROM dup_mitglieder WHERE gruppe_id=?)", (gruppe_id,)).fetchone()
    veraendert = con.execute(
        "SELECT 1 FROM dup_mitglieder m JOIN lieder l ON l.id=m.lied_id WHERE m.gruppe_id=? "
        "AND l.status!='analysiert'", (gruppe_id,)).fetchone()
    neu = "ersetzt" if ueberschneidung or veraendert else "offen"
    con.execute("UPDATE dup_gruppen SET status=?, entschieden_von=NULL WHERE id=?", (neu, gruppe_id))
    protokoll(con, "duplikat_zuruecknehmen", {"gruppe": gruppe_id, "war": r[0], "von": r[1], "jetzt": neu}, wer)
    return neu


if __name__ == "__main__":
    finden()
