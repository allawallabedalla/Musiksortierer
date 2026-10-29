"""Gesangssprache je Lied: Deutschrap und US-Rap klingen gleich, der Klang-Steckbrief trennt sie nicht.

Whisper (Modell „tiny“, nur die Spracherkennung, keine Abschrift) hört zwei 30-s-Stellen: den Ausschnitt fürs
Bewerten und die Liedmitte (ein Intro oder Skit allein täuscht). Die Wahrscheinlichkeiten werden gemittelt.
Test 29.10. an 145 bewerteten Hip-Hop-Liedern: deutsch 61/62 erkannt (der fehlende ist ein Instrumental),
0/83 fälschlich deutsch; ~1,7 s je Lied auf zwei Kernen.

Gespeichert in lieder: sprache (Kürzel der wahrscheinlichsten, „fehler“ wenn nicht lesbar), sprache_p,
sprache_probs (JSON, die 8 wahrscheinlichsten). Instrumentals haben keine klare Sprache (sprache_p niedrig).

  python -m musiksort.sprache [--anzahl N]   # noch nicht gemessene Lieder messen (bewertete zuerst)
  python -m musiksort.sprache --zaehlen
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np

from . import config, db

# Eigene Spalten im Genre-Modell; alle anderen Sprachen zusammen in „andere“
SPRACHEN = ("de", "en", "fr", "es", "it", "tr")
NAMEN = {"de": "Deutsch", "en": "Englisch", "fr": "Französisch", "es": "Spanisch", "it": "Italienisch",
         "tr": "Türkisch", "nl": "Niederländisch", "pt": "Portugiesisch", "ru": "Russisch", "pl": "Polnisch",
         "ja": "Japanisch", "ko": "Koreanisch", "zh": "Chinesisch", "ar": "Arabisch", "sv": "Schwedisch",
         "hi": "Hindi", "el": "Griechisch", "la": "Latein", "jw": "Javanisch"}
# ISO 639-2 für die Datei-Tags (ID3 TLAN & Co.)
ISO = {"de": "deu", "en": "eng", "fr": "fra", "es": "spa", "it": "ita", "tr": "tur", "nl": "nld", "pt": "por",
       "ru": "rus", "pl": "pol", "ja": "jpn", "ko": "kor", "zh": "zho", "ar": "ara", "sv": "swe", "hi": "hin",
       "el": "ell", "la": "lat"}
BREITE = len(SPRACHEN) + 2  # + andere + gemessen

_modell = None


def modell():
    global _modell
    if _modell is None:
        from faster_whisper import WhisperModel
        _modell = WhisperModel(str(config.WHISPER), device="cpu", compute_type="int8", cpu_threads=2)
    return _modell


def _audio(pfad, start, dauer=30):
    roh = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.1f}", "-t", str(dauer), "-i", str(pfad),
                          "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True,
                         timeout=120).stdout
    return np.frombuffer(roh, np.float32)


def stellen(dauer, snippet_start):
    """Ausschnitt und Liedmitte; bei kurzen Liedern nur eine Stelle."""
    d = dauer or 0
    a = max(0.0, min(snippet_start or 0, max(0.0, d - 30)))
    mitte = max(0.0, d / 2 - 15)
    return [a] if d < 75 or abs(mitte - a) < 20 else [a, mitte]


def messen(pfad, dauer, snippet_start):
    """{sprache: wahrscheinlichkeit} (die 8 wahrscheinlichsten, gemittelt über die Stellen)."""
    summe = {}
    st = stellen(dauer, snippet_start)
    for s in st:
        x = _audio(pfad, s)
        if len(x) < 16000 * 5:
            continue
        _, _, alle = modell().detect_language(x)
        for code, p in alle:
            summe[code] = summe.get(code, 0.0) + p / len(st)
    if not summe:
        raise ValueError("kein Audio")
    return dict(sorted(summe.items(), key=lambda kv: -kv[1])[:8])


def merkmale(probs_json):
    """Spalten fürs Genre-Modell aus sprache_probs (JSON oder None): je SPRACHEN, andere, gemessen."""
    erg = np.zeros((len(probs_json), BREITE), dtype=np.float32)
    for i, j in enumerate(probs_json):
        if not j:
            continue
        try:
            p = json.loads(j)
        except (TypeError, ValueError):
            continue
        bekannt = [float(p.get(s, 0.0)) for s in SPRACHEN]
        erg[i, :len(SPRACHEN)] = bekannt
        erg[i, len(SPRACHEN)] = max(0.0, sum(float(v) for v in p.values()) - sum(bekannt))
        erg[i, -1] = 1.0
    return erg * config.SPRACHE_GEWICHT


def anzeige(sprache, p):
    """Für die Karte: „Deutsch“, „wohl Englisch“, None (unklar/instrumental/nicht gemessen)."""
    if not sprache or sprache == "fehler" or p is None or p < 0.5:
        return None
    name = NAMEN.get(sprache, sprache)
    return name if p >= config.SPRACHE_DATEI_P else f"wohl {name}"


def _zuerst(con):
    """Genre-IDs, deren Lieder zuerst gemessen werden (Teilbäume von SPRACHE_ZUERST)."""
    from .lernen import _baum
    teil = _baum(con)[0]
    ids = set()
    for (gid,) in con.execute(f"SELECT id FROM genres WHERE name IN ({','.join('?' * len(config.SPRACHE_ZUERST))})",
                              config.SPRACHE_ZUERST):
        ids |= teil.get(gid, {gid})
    return ids


def offen(con, anzahl=None):
    """Noch nicht gemessene Lieder: bewertete zuerst, dann was das Modell bei SPRACHE_ZUERST sieht, dann der Rest."""
    zuerst = sorted(_zuerst(con)) or [-1]
    platz = ",".join("?" * len(zuerst))
    sql = f"""SELECT l.id, l.pfad, l.dauer, l.snippet_start FROM lieder l LEFT JOIN vorhersagen v ON v.lied_id=l.id
              WHERE l.status='analysiert' AND l.sprache IS NULL
              ORDER BY EXISTS(SELECT 1 FROM etiketten e WHERE e.lied_id=l.id) DESC,
                       (v.genre_id IN ({platz}) OR v.ober_id IN ({platz})) DESC, l.id"""
    if anzahl:
        sql += f" LIMIT {int(anzahl)}"
    return con.execute(sql, zuerst + zuerst).fetchall()


def laufen(con=None, anzahl=None, ausgabe=print):
    con = con or db.verbinden()
    liste = offen(con, anzahl)
    ausgabe(f"{len(liste)} Lieder ohne Sprache")
    t0, n, fehler = time.time(), 0, 0
    for r in liste:
        try:
            probs = messen(config.medienpfad(r["pfad"]), r["dauer"], r["snippet_start"])
            code, p = next(iter(probs.items()))
            werte = (code, round(p, 4), json.dumps({k: round(v, 4) for k, v in probs.items()}))
        except Exception as e:  # unlesbar: nicht jede Nacht neu versuchen
            fehler += 1
            ausgabe(f"{r['pfad']}: {e!r}"[:300])
            werte = ("fehler", None, None)
        with con:
            con.execute("UPDATE lieder SET sprache=?, sprache_p=?, sprache_probs=?, sprache_ts=? WHERE id=?",
                        (*werte, db.jetzt(), r["id"]))
        n += 1
        if n % 200 == 0:
            ausgabe(f"{n}/{len(liste)} · {(time.time() - t0) / n:.1f} s je Lied · {fehler} Fehler")
    ausgabe(f"fertig: {n} gemessen, {fehler} Fehler")
    return n


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--anzahl", type=int)
    a.add_argument("--zaehlen", action="store_true")
    args = a.parse_args()
    c = db.verbinden()
    if args.zaehlen:
        print(c.execute("SELECT count(*) FROM lieder WHERE status='analysiert' AND sprache IS NULL").fetchone()[0],
              "Lieder ohne Sprache")
        sys.exit(0)
    laufen(c, args.anzahl)
