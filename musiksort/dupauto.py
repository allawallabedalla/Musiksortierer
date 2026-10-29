"""Duplikatgruppen automatisch entscheiden, in Stufen vom Sichersten zum Unsichersten.

Jede Entscheidung läuft durch dieselben Prüfungen wie ein Klick (duplikate.entscheiden),
steht im Protokoll als „automatisch“ mit Grund und lässt sich zurücknehmen
(einzeln oder alle). Gelöscht wird nichts: das macht erst der Sortierlauf.

  1  Bitgleich: alle Dateien der Gruppe haben dieselbe Prüfsumme (SHA-256 über die ganze Datei)
  2  Eng: gleiche Aufnahme laut Fingerabdruck (≤ 5 % Abweichung, Länge ohne Stille ≤ 1 s) und auch
     das Ende (letzte 2 Minuten bis zum letzten Hörbaren) stimmt → beste Qualität behalten
  C  Namen: die Namen nennen verschiedene Fassungen (Radio Edit, Live, Remix …) → alle behalten;
     ebenso Zweifelsfälle, deren Länge um > 8 s abweicht
  A  Ende: auch der Fingerabdruck der letzten 2 Minuten (ohne Stille am Ende) stimmt → beste behalten
  B  AcoustID: gleiche Aufnahme-Kennung → beste behalten; ganz verschiedene oder Titel nennen
     verschiedene Fassungen (Radio Edit, Extended …) → alle behalten
     (braucht ACOUSTID_KEY in ~/.env; schickt Fingerabdrücke an acoustid.org)
  D  Lernen: ein Modell aus deinen eigenen Entscheidungen; entscheidet erst ab 200 davon und
     nur, wo es in der Kreuzvalidierung vorsichtig gerechnet ≥ 98 % so entscheidet wie du

  python -m musiksort.dupauto --trocken        # nur zählen
  python -m musiksort.dupauto [--stufen 1,2,C,A,B,D]
  python -m musiksort.dupauto --nachpruefen 2  # frühere Entscheidungen einer Stufe mit den heutigen Regeln prüfen

„Beste Qualität“: verlustfrei vor Bitrate; eine FLAC/WAV mit Kodierer-Kante unter ~19,5 kHz (aufgeblasene
MP3) zählt wie die MP3, die sie war (klang.py, Grenzfrequenz wird vor jeder Entscheidung gemessen).
"""
import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import time

import numpy as np

from . import config, db, duplikate, klang

STUFEN = ["1", "2", "C", "A", "B", "D"]
ENG_ABWEICHUNG, ENG_DAUER = 0.05, 1.0
ZWEIFEL_LAENGE = 8.0
ENDE_SEKUNDEN = 120
ACOUSTID_SCORE = 0.8

# Wörter, an denen man eine andere Fassung erkennt
FASSUNG = re.compile(r"\b(radio|edit|extended|live|remaster(?:ed)?|remix|rmx|mix|version|instrumental|acoustic|"
                     r"unplugged|dub|vip|bootleg|mashup|club|original|demo|karaoke|a\s?cappella|reprise)\b", re.I)


def fassungs_woerter(pfad, titel):
    name = (titel or "") + " " + pfad.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return frozenset(w.lower().replace(" ", "") for w in FASSUNG.findall(name))


def offene_gruppen(con):
    """Offene, noch nicht durch eine andere Freigabe erledigte Gruppen mit allen nötigen Angaben."""
    from .duplikate import OFFENE_GRUPPEN
    gruppen = collections.defaultdict(list)
    sicher = {}
    for r in con.execute(
            f"SELECT g.id gid, g.sicher, m.abweichung, l.* FROM dup_gruppen g JOIN dup_mitglieder m ON m.gruppe_id=g.id "
            f"JOIN lieder l ON l.id=m.lied_id WHERE {OFFENE_GRUPPEN} AND l.status='analysiert' ORDER BY g.id"):
        gruppen[r["gid"]].append(dict(r))
        sicher[r["gid"]] = bool(r["sicher"])
    return [(gid, sicher[gid], ms) for gid, ms in gruppen.items() if len(ms) >= 2]


def beste(ms):
    return max(ms, key=duplikate.qualitaet)


def beste_gemessen(con, ms):
    """Beste Datei, nachdem jede „verlustfreie“ auf eine Kodierer-Kante geprüft ist (falsche FLACs)."""
    for m in ms:
        if m["verlustfrei"]:
            klang.grenzfrequenz(con, m)
    return beste(ms)


def eng(con, ms):
    """≤ 5 % Abweichung und Länge ≤ 1 s; bei 1–8 s Unterschied zählt die Länge ohne Stille."""
    if max(m["abweichung"] or 0 for m in ms) > ENG_ABWEICHUNG:
        return False
    dauer = [m["dauer"] or 0 for m in ms]
    if max(dauer) - min(dauer) <= ENG_DAUER:
        return True
    if max(dauer) - min(dauer) > ZWEIFEL_LAENGE:
        return False
    for m in ms:
        klang.stille(con, m)
    netto = [klang.netto_dauer(m) for m in ms]
    return max(netto) - min(netto) <= ENG_DAUER


def ende_gleich(con, ms, b):
    ende_b = klang.fp_ende(con, b)
    return all(m is b or duplikate.bitfehler(ende_b, klang.fp_ende(con, m)) <= duplikate.MAX_BITFEHLER for m in ms)


# ---------------------------------------------------------------- Lesen vom NAS

def sha256(con, m):
    """Prüfsumme über die ganze Datei; zwischengespeichert, solange Größe und Änderungszeit gleich sind."""
    stand = f"{m['groesse']}:{m['mtime']}"
    if m.get("sha256") and m.get("sha_stand") == stand:
        return m["sha256"]
    h = hashlib.sha256()
    with open(config.medienpfad(m["pfad"]), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    m["sha256"], m["sha_stand"] = h.hexdigest(), stand
    with con:
        con.execute("UPDATE lieder SET sha256=?, sha_stand=? WHERE id=?", (m["sha256"], stand, m["id"]))
    return m["sha256"]


def acoustid_aufnahmen(con, m, schluessel, nachschlagen):
    """{Aufnahme-Kennung: Score} laut AcoustID (zwischengespeichert, mit Titeln: acoustid_titel)."""
    return _acoustid(con, m, schluessel, nachschlagen)["aufn"]


def acoustid_titel(con, m, schluessel, nachschlagen):
    """{Aufnahme-Kennung: offizieller Titel} laut AcoustID/MusicBrainz."""
    return _acoustid(con, m, schluessel, nachschlagen)["titel"]


def _acoustid(con, m, schluessel, nachschlagen):
    if m.get("acoustid"):
        gespeichert = json.loads(m["acoustid"])
        if gespeichert.get("v") == 2:
            return gespeichert
    import chromaprint
    roh = np.frombuffer(m["fp"], np.uint32).astype(np.int64)
    roh = [int(x) - (1 << 32) if x >= (1 << 31) else int(x) for x in roh]  # Chromaprint erwartet int32
    kodiert = chromaprint.encode_fingerprint(roh, 1)
    antwort = nachschlagen(schluessel, kodiert, int(round(m["dauer"] or 0)), meta="recordings")
    if antwort.get("status") not in (None, "ok"):
        raise RuntimeError(f"AcoustID: {antwort.get('error')}")
    aufnahmen, titel = {}, {}
    for treffer in antwort.get("results", []):
        for rec in treffer.get("recordings", []) or []:
            aufnahmen[rec["id"]] = max(aufnahmen.get(rec["id"], 0), float(treffer.get("score", 0)))
            if rec.get("title"):
                titel[rec["id"]] = rec["title"]
    erg = {"v": 2, "aufn": aufnahmen, "titel": titel}
    m["acoustid"] = json.dumps(erg, ensure_ascii=False)
    with con:
        con.execute("UPDATE lieder SET acoustid=? WHERE id=?", (m["acoustid"], m["id"]))
    time.sleep(0.34)  # höchstens 3 Anfragen je Sekunde (Regel von acoustid.org)
    return erg


# ---------------------------------------------------------------- Stufen
# Jede Stufe liefert für eine Gruppe None (nicht zuständig/unsicher) oder (status, behalten, grund).

def stufe_1(con, sicher, ms, **_):
    if not sicher or len({m["groesse"] for m in ms}) != 1:
        return None
    if len({sha256(con, m) for m in ms}) != 1:
        return None
    return "freigegeben", beste(ms)["id"], "Prüfsumme identisch"  # bitgleich: Qualität ist gleich


def stufe_2(con, sicher, ms, **_):
    if not (sicher and eng(con, ms)):
        return None
    b = beste_gemessen(con, ms)
    if not ende_gleich(con, ms, b):
        return None
    return "freigegeben", b["id"], "gleiche Aufnahme, Länge ohne Stille ≤ 1 s, Abweichung ≤ 5 %, Ende gleich"


def stufe_c(con, sicher, ms, **_):
    if len({fassungs_woerter(m["pfad"], m["titel"]) for m in ms}) > 1:
        return "verworfen", beste(ms)["id"], "Namen nennen verschiedene Fassungen"
    dauer = [m["dauer"] or 0 for m in ms]
    if not sicher and max(dauer) - min(dauer) > ZWEIFEL_LAENGE:
        return "verworfen", beste(ms)["id"], f"Länge weicht um > {ZWEIFEL_LAENGE:.0f} s ab"
    return None


def stufe_a(con, sicher, ms, **_):
    if not sicher:  # Zweifelsfälle (Remaster?) nie über das Ende allein
        return None
    b = beste_gemessen(con, ms)
    if not ende_gleich(con, ms, b):
        return None
    return "freigegeben", b["id"], "Anfang und Ende gleich (ohne Stille)"


def stufe_b(con, sicher, ms, schluessel=None, nachschlagen=None, **_):
    if not schluessel:
        return None
    sets = []
    for m in ms:
        aufn = {k for k, v in acoustid_aufnahmen(con, m, schluessel, nachschlagen).items() if v >= ACOUSTID_SCORE}
        if not aufn:
            return None  # AcoustID kennt eine der Dateien nicht sicher genug
        sets.append(aufn)
    gemeinsam = set.intersection(*sets)
    dauer = [m["dauer"] or 0 for m in ms]
    if gemeinsam and (sicher or max(dauer) - min(dauer) <= 3):
        return "freigegeben", beste_gemessen(con, ms)["id"], "AcoustID: gleiche Aufnahme"
    if not gemeinsam and all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:]):
        return "verworfen", beste(ms)["id"], "AcoustID: verschiedene Aufnahmen"
    # Titel der wahrscheinlichsten Aufnahme je Datei: nennen sie verschiedene Fassungen?
    fassungen = set()
    for m in ms:
        aufn = acoustid_aufnahmen(con, m, schluessel, nachschlagen)
        titel = acoustid_titel(con, m, schluessel, nachschlagen)
        top = max(aufn, key=aufn.get)
        if not titel.get(top):
            return None  # Titel unbekannt: daraus nichts schließen
        fassungen.add(fassungs_woerter("", titel[top]))
    if len(fassungen) > 1:
        return "verworfen", beste(ms)["id"], "AcoustID: Titel nennen verschiedene Fassungen"
    return None


def merkmale(sicher, ms):
    dauer = [m["dauer"] or 0 for m in ms]
    raten = [m["bitrate"] or 0 for m in ms if not m["verlustfrei"]]
    ordner = {m["pfad"].split("/", 1)[0] for m in ms}
    return [float(sicher), max(m["abweichung"] or 0 for m in ms), max(dauer) - min(dauer),
            (max(dauer) - min(dauer)) / max(max(dauer), 1), len(ms),
            (min(raten) / max(raten)) if raten and max(raten) else 1.0, float(any(m["verlustfrei"] for m in ms)),
            float(len({fassungs_woerter(m["pfad"], m["titel"]) for m in ms}) > 1),
            float(len({m["groesse"] for m in ms}) == 1), float(bool(ordner & config.DUPLIKAT_ORDNER)),
            float(bool(ordner & config.BACKUP_ORDNER))]


class Lernmodell:
    """Stufe D: sagt voraus, ob du „beste behalten“ (1) oder „alle behalten“ (0) wählen würdest."""

    def __init__(self, con):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from .lernen import schwelle_finden
        self.modell, self.schwelle = None, {}
        X, y = [], []
        gruppen = collections.defaultdict(list)
        for r in con.execute("SELECT g.id gid, g.sicher, g.status, m.abweichung, m.behalten, l.* FROM dup_gruppen g "
                             "JOIN dup_mitglieder m ON m.gruppe_id=g.id JOIN lieder l ON l.id=m.lied_id "
                             "WHERE g.entschieden_von='ich' AND g.status IN ('freigegeben','verworfen','erledigt')"):
            gruppen[r["gid"]].append(dict(r))
        for ms in gruppen.values():
            if ms[0]["status"] in ("freigegeben", "erledigt"):
                behalten = next(m for m in ms if m["behalten"])
                if behalten["id"] != beste(ms)["id"]:
                    continue  # du hast bewusst eine andere Datei gewählt: lernt das Modell nicht
                y.append(1)
            else:
                y.append(0)
            X.append(merkmale(bool(ms[0]["sicher"]), ms))
        self.n = len(y)
        if self.n < config.DUP_AUTO_D_MIN or len(set(y)) < 2 or min(np.bincount(y)) < 5:
            return
        X, y = np.array(X), np.array(y)
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
        vorh = p.argmax(1)
        for k in (0, 1):
            s = schwelle_finden(p[vorh == k, k], y[vorh == k] == k, ziel=config.DUP_AUTO_D_ZIEL)
            if s:
                self.schwelle[k] = s[0]
        self.modell = m.fit(X, y) if self.schwelle else None

    def __call__(self, con, sicher, ms, **_):
        if self.modell is None:
            return None
        p = self.modell.predict_proba([merkmale(sicher, ms)])[0]
        k = int(p.argmax())
        if k not in self.schwelle or p[k] < self.schwelle[k]:
            return None
        return ("freigegeben" if k else "verworfen"), beste(ms)["id"], f"gelernt aus deinen Entscheidungen ({p[k]:.0%})"


# ---------------------------------------------------------------- Ablauf

def laufen(con=None, stufen=STUFEN, trocken=False, nachschlagen=None, ausgabe=print):
    con = con or db.verbinden()
    schluessel = config.env("ACOUSTID_KEY")
    if nachschlagen is None and schluessel:
        import acoustid
        nachschlagen = lambda k, fp, d, meta: acoustid.lookup(k, fp, d, meta=meta)  # noqa: E731
    pruefer = {"1": stufe_1, "2": stufe_2, "C": stufe_c, "A": stufe_a, "B": stufe_b}
    if "D" in stufen:
        pruefer["D"] = Lernmodell(con)
        ausgabe(f"Stufe D: {pruefer['D'].n} eigene Entscheidungen, "
                + ("entscheidet mit" if pruefer["D"].modell else f"noch nicht aktiv (ab {config.DUP_AUTO_D_MIN})"))
    if "B" in stufen and not schluessel:
        ausgabe("Stufe B: kein ACOUSTID_KEY in ~/.env, übersprungen")
    zaehler = collections.Counter()
    erledigt = set()  # im Trockenlauf: was eine frühere Stufe schon nähme
    for stufe in stufen:
        for gid, sicher, ms in offene_gruppen(con):
            if gid in erledigt:
                continue
            try:
                urteil = pruefer[stufe](con, sicher, ms, schluessel=schluessel, nachschlagen=nachschlagen)
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
                zaehler[(stufe, "Lesefehler")] += 1
                if not config.MOUNT_PRUEFEN or __import__("musiksort.scan", fromlist=["nas_da"]).nas_da():
                    continue
                ausgabe(f"NAS weg ({e}): Abbruch")
                return zaehler
            except Exception as e:  # z. B. AcoustID nicht erreichbar: diese Gruppe auslassen
                zaehler[(stufe, "Fehler")] += 1
                ausgabe(f"Stufe {stufe}, Gruppe {gid}: {e!r}")
                continue
            if urteil is None:
                continue
            status, behalten, grund = urteil
            if trocken:
                erledigt.add(gid)
                zaehler[(stufe, status)] += 1
                continue
            con.execute("BEGIN IMMEDIATE")
            try:
                duplikate.entscheiden(con, gid, status, [m["id"] for m in ms], behalten,
                                      von=f"auto:{stufe}", wer="dupauto", grund=grund)
                con.commit()
                zaehler[(stufe, status)] += 1
            except duplikate.Abgelehnt:
                con.rollback()
                zaehler[(stufe, "Konflikt")] += 1
            except BaseException:
                con.rollback()
                raise
        ausgabe(f"Stufe {stufe}: " + (", ".join(f"{k[1]} {v}" for k, v in sorted(zaehler.items()) if k[0] == stufe) or "–"))
    return zaehler


def nachpruefen(con=None, stufe="2", ausgabe=lambda s: print(s, flush=True)):
    """Frühere automatische Entscheidungen einer Stufe mit den heutigen Regeln prüfen.

    Hält eine Entscheidung nicht mehr (Ende passt nicht, andere Datei wäre die beste), wird sie
    zurückgenommen; die Gruppe wird bei der nächsten Duplikatsuche neu vorgelegt bzw. entschieden.
    Messungen landen im Zwischenspeicher: ein abgebrochener Lauf setzt beim nächsten Mal schnell fort.
    """
    con = con or db.verbinden()
    pruefer = {"2": stufe_2, "A": stufe_a}[stufe]
    gruppen = collections.defaultdict(list)
    sicher = {}
    for r in con.execute("SELECT g.id gid, g.sicher, m.abweichung, m.behalten, l.* FROM dup_gruppen g "
                         "JOIN dup_mitglieder m ON m.gruppe_id=g.id JOIN lieder l ON l.id=m.lied_id "
                         "WHERE g.entschieden_von=? AND g.status='freigegeben' ORDER BY g.id", (f"auto:{stufe}",)):
        gruppen[r["gid"]].append(dict(r))
        sicher[r["gid"]] = bool(r["sicher"])
    z = collections.Counter()
    for i, (gid, ms) in enumerate(gruppen.items(), 1):
        try:
            urteil = pruefer(con, sicher[gid], ms)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            z["Lesefehler"] += 1
            if config.MOUNT_PRUEFEN and not __import__("musiksort.scan", fromlist=["nas_da"]).nas_da():
                ausgabe("NAS weg: Abbruch")
                break
            continue
        behalten = next(m["id"] for m in ms if m["behalten"])
        if urteil and urteil[0] == "freigegeben" and urteil[1] == behalten:
            z["hält"] += 1
        else:
            grund = "Regel passt nicht mehr" if not urteil else "andere Datei ist die beste"
            con.execute("BEGIN IMMEDIATE")
            try:
                duplikate.zuruecknehmen(con, gid, wer="dupauto-nachpruefen")
                duplikate.protokoll(con, "duplikat_nachpruefung", {"gruppe": gid, "stufe": stufe, "grund": grund},
                                    "dupauto-nachpruefen")
                con.commit()
                z["zurückgenommen: " + grund] += 1
            except BaseException:
                con.rollback()
                raise
        if i % 200 == 0:
            ausgabe(f"{i}/{len(gruppen)} {dict(z)}")
    ausgabe(f"Nachprüfung Stufe {stufe}: {dict(z)}")
    return z


def alle_zuruecknehmen(con, stufe=None, wer=None):
    """Automatische Entscheidungen zurücknehmen (alle oder nur eine Stufe). Rückgabe: Anzahl."""
    muster = f"auto:{stufe}" if stufe else "auto:%"
    ids = [r[0] for r in con.execute("SELECT id FROM dup_gruppen WHERE entschieden_von LIKE ? "
                                     "AND status IN ('freigegeben','verworfen')", (muster,))]
    for gid in ids:
        duplikate.zuruecknehmen(con, gid, wer=wer)
    return len(ids)


def uebersicht(con):
    """{stufe: {freigegeben: n, verworfen: n}} der automatischen Entscheidungen, die gerade gelten."""
    erg = collections.defaultdict(dict)
    for von, status, n in con.execute("SELECT entschieden_von, status, COUNT(*) FROM dup_gruppen "
                                      "WHERE entschieden_von LIKE 'auto:%' AND status IN ('freigegeben','verworfen') "
                                      "GROUP BY 1, 2"):
        erg[von.split(":", 1)[1]][status] = n
    return dict(erg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trocken", action="store_true", help="nur zählen, nichts entscheiden")
    ap.add_argument("--stufen", default=",".join(STUFEN))
    ap.add_argument("--nachpruefen", choices=["2", "A"], help="frühere Entscheidungen dieser Stufe neu prüfen")
    a = ap.parse_args()
    if a.nachpruefen:
        nachpruefen(stufe=a.nachpruefen)
        sys.exit(0)
    stufen = [s.strip().upper() for s in a.stufen.split(",") if s.strip()]
    if not set(stufen) <= set(STUFEN):
        sys.exit(f"Stufen: {', '.join(STUFEN)}")
    z = laufen(stufen=stufen, trocken=a.trocken)
    print("trocken:" if a.trocken else "entschieden:", dict(sorted(((f"{k[0]} {k[1]}"), v) for k, v in z.items())))
