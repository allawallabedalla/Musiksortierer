"""Einlesen: Dateien finden, dann je Lied in EINEM Dekodierdurchgang
Tags, Qualität, Fingerabdruck, Snippet-Start, BPM und Klang-Steckbrief.

  python -m musiksort.scan suchen              # Dateiliste aktualisieren
  python -m musiksort.scan analysieren [-n 500] [--zufall] [-j 2]
"""
import argparse
import json
import multiprocessing
import os
import random
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import numpy as np

from . import config, db


# ---------------------------------------------------------------- Suchen

def _mit_wiederholung(f, *args, versuche=4):
    """SMB liefert unter Last gelegentlich kurzzeitig ENOENT/EIO."""
    for i in range(versuche):
        try:
            return f(*args)
        except OSError:
            if i == versuche - 1:
                raise
            time.sleep(0.5 * 2 ** i)


def _durchlaufen(wurzel, fehler):
    stapel = [str(wurzel)]
    while stapel:
        ordner = stapel.pop()
        try:
            eintraege = _mit_wiederholung(lambda o: list(os.scandir(o)), ordner)
        except OSError as e:
            print(f"übersprungen: {ordner}: {e}", file=sys.stderr)
            fehler.append(ordner)
            continue
        for e in eintraege:
            if e.name.startswith((".", "._")):
                continue
            try:
                ist_ordner = e.is_dir(follow_symlinks=False)
            except OSError:
                fehler.append(e.path)  # sonst gälte alles darunter als „fehlt“
                continue
            if ist_ordner:
                if config.ausgeschlossen(os.path.relpath(e.path, wurzel)):
                    continue
                if e.name in ("@eaDir", "#recycle"):
                    continue
                stapel.append(e.path)
            elif os.path.splitext(e.name)[1].lower() in config.AUDIO_ENDUNGEN:
                try:
                    st = _mit_wiederholung(e.stat)
                except OSError:
                    fehler.append(e.path)
                    continue
                yield e.path, st.st_size, st.st_mtime


def suchen():
    con = db.verbinden()
    wurzel = config.MEDIA_ROOT
    if not nas_da():
        raise SystemExit(f"{wurzel} ist nicht eingebunden; nichts geändert")
    bekannt = {r["pfad"]: (r["groesse"], r["mtime"], r["status"], r["emb"] is not None)
               for r in con.execute("SELECT pfad, groesse, mtime, status, emb FROM lieder")}
    # ohne Einstellungen fehlen die eigenen Ausschlüsse: bei bestehender Sammlung nicht still anders suchen
    if not config.EINSTELLUNGEN_DA and bekannt:
        raise SystemExit(f"{config.EINSTELLUNGEN} fehlt; ohne die eigenen Ordner-Einstellungen wird nicht gesucht")
    gesehen, neu, geaendert, zurueck, fehler, puffer = set(), 0, 0, 0, [], []

    def schreiben():
        # kurze Transaktionen: die App darf zwischendurch bewerten
        if puffer:
            with con:
                for sql, werte in puffer:
                    con.execute(sql, werte)
            puffer.clear()

    for pfad, groesse, mtime in _durchlaufen(wurzel, fehler):
        try:
            rel = os.path.relpath(pfad, wurzel)
            rel.encode("utf-8")
        except (ValueError, UnicodeError):
            fehler.append(pfad)
            continue
        if rel in gesehen:  # SMB liefert unter Last gelegentlich denselben Eintrag doppelt
            continue
        gesehen.add(rel)
        alt = bekannt.get(rel)
        if alt is None:
            puffer.append(("INSERT INTO lieder(pfad, groesse, mtime) VALUES (?,?,?)", (rel, groesse, mtime)))
            neu += 1
        elif alt[:2] != (groesse, mtime):
            puffer.append(("UPDATE lieder SET groesse=?, mtime=?, status='neu', versuche=0 WHERE pfad=?",
                           (groesse, mtime, rel)))
            geaendert += 1
        elif alt[2] in ("fehlt", "ausgeschlossen", "geloescht"):  # wieder da (bzw. nicht mehr ausgeschlossen) und unverändert
            puffer.append(("UPDATE lieder SET status=? WHERE pfad=?", ("analysiert" if alt[3] else "neu", rel)))
            zurueck += 1
        if len(puffer) >= 500:
            schreiben()
    schreiben()

    # Nicht mehr gefunden: als „fehlt“ markieren, nicht löschen (Bewertungen bleiben).
    # Unter Ordnern, die gerade nicht lesbar waren, wird nichts markiert.
    if str(wurzel) in fehler or not nas_da():
        raise SystemExit("Wurzel nicht lesbar: nichts als „fehlt“ markiert")
    unlesbar = [os.path.relpath(f, wurzel) + os.sep for f in fehler]
    weg = [p for p in set(bekannt) - gesehen
           if bekannt[p][2] not in ("fehlt", "ausgeschlossen", "papierkorb", "geloescht")
           and not any(p.startswith(u) for u in unlesbar)]
    for i in range(0, len(weg), 500):
        with con:
            con.executemany("UPDATE lieder SET status='fehlt' WHERE pfad=?", [(p,) for p in weg[i:i + 500]])

    # Ordner, die nachträglich ausgeschlossen wurden: als „ausgeschlossen“ markieren (nie löschen:
    # sonst würden IDs frei und könnten wiederverwendet werden) und aus allem Weiteren herausnehmen
    with con:
        for ordner in sorted(config.AUSGESCHLOSSEN | config.AUSGESCHLOSSENE_PFADE):
            muster = ordner.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "/%"
            ids = "SELECT id FROM lieder WHERE pfad LIKE ? ESCAPE '\\' AND status!='ausgeschlossen'"
            for tabelle in ("etiketten", "uebersprungen", "vorhersagen", "stichproben", "klanggruppen"):
                con.execute(f"DELETE FROM {tabelle} WHERE lied_id IN ({ids})", (muster,))
            con.execute(f"UPDATE lieder SET status='ausgeschlossen' WHERE id IN ({ids})", (muster,))
        # Duplikat-Entscheidungen, bei denen eine Datei nicht mehr „analysiert“ ist (geändert,
        # verschwunden, ausgeschlossen), gelten nicht mehr: neu prüfen statt alte Freigabe ausführen
        n_ungueltig = gruppen_ungueltig_machen(con)
    print(f"neu {neu}, geändert {geaendert}, wieder da {zurueck}, nicht mehr gefunden {len(weg)}, "
          f"gesamt {len(gesehen)}, unlesbar {len(fehler)}, Duplikatgruppen ungültig {n_ungueltig}")


def gruppen_ungueltig_machen(con):
    """Gruppen mit einem Mitglied, das nicht (mehr) „analysiert“ ist, auf „ersetzt“ setzen."""
    return con.execute(
        "UPDATE dup_gruppen SET status='ersetzt' WHERE status NOT IN ('ersetzt','erledigt') AND id IN ("
        "SELECT m.gruppe_id FROM dup_mitglieder m JOIN lieder l ON l.id=m.lied_id WHERE l.status!='analysiert')"
    ).rowcount


def nas_da():
    """Ist die Freigabe eingebunden und lesbar? Ohne sie wäre jede Datei ein „Fehler“."""
    try:
        return (os.path.ismount(config.MEDIA_ROOT) or not config.MOUNT_PRUEFEN) and bool(os.listdir(config.MEDIA_ROOT))
    except OSError:
        return False


# ---------------------------------------------------------------- Analysieren

_modell = {}
MAX_VERSUCHE = 3
# Meldungen, hinter denen ein NAS-Aussetzer stecken kann (nicht die Datei selbst)
LESEFEHLER = ("No such file", "Input/output error", "Host is down", "Stale file handle",
              "Resource temporarily unavailable", "Connection timed out", "Permission denied")


class Voruebergehend(Exception):
    """Lesefehler, der am NAS liegen kann (Aussetzer): Lied bleibt „neu“ und wird erneut versucht."""


def _init_worker():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    from essentia.standard import (PercivalBpmEstimator, TensorflowPredict2D,
                                   TensorflowPredictEffnetDiscogs)
    _modell["effnet"] = TensorflowPredictEffnetDiscogs(graphFilename=str(config.EFFNET), output="PartitionedCall:1")
    _modell["kopf"] = TensorflowPredict2D(graphFilename=str(config.DISCOGS400),
                                          input="serving_default_model_Placeholder", output="PartitionedCall:0")
    _modell["bpm"] = PercivalBpmEstimator(sampleRate=config.SR)
    _modell["klassen"] = json.load(open(config.DISCOGS400_JSON))["classes"]


def _ffprobe(pfad):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", pfad],
                       capture_output=True, text=True, timeout=60)
    if r.returncode:
        meldung = r.stderr.strip()[:200]
        if any(t in meldung for t in LESEFEHLER):
            raise Voruebergehend("ffprobe: " + meldung)  # liegt vermutlich am NAS, nicht an der Datei
        raise RuntimeError("ffprobe: " + meldung)
    d = json.loads(r.stdout)
    spuren = d.get("streams") or []
    s = next((x for x in spuren if x.get("codec_type") == "audio"), {})
    f = d.get("format", {})
    tags = {k.lower(): v for k, v in {**f.get("tags", {}), **s.get("tags", {})}.items()}
    codec = s.get("codec_name", "")
    if codec.startswith("pcm_"):
        codec = "aiff" if f.get("format_name", "").startswith("aiff") else "wav"
    bittiefe = int(s.get("bits_per_raw_sample") or s.get("bits_per_sample") or 0) or None
    return {
        "codec": codec,
        "bitrate": int(s.get("bit_rate") or f.get("bit_rate") or 0) // 1000 or None,
        "samplerate": int(s.get("sample_rate") or 0) or None,
        "bittiefe": bittiefe,
        "kanaele": s.get("channels"),
        "dauer": float(f.get("duration") or s.get("duration") or 0) or None,
        "verlustfrei": int(codec in config.VERLUSTFREI),
        "interpret": tags.get("artist") or tags.get("album_artist"),
        "titel": tags.get("title"),
        "album": tags.get("album"),
        "tag_genre": tags.get("genre"),
        # echte Videospur (kein eingebettetes Cover): Handyclip, Vortrag, Musikvideo – kein Lied
        "video": any(x.get("codec_type") == "video" and not (x.get("disposition") or {}).get("attached_pic")
                     for x in spuren),
    }


def _dekodieren(pfad, dauer=None):
    r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", pfad, "-t", str(config.MAX_ANALYSE_S),
                        "-vn", "-ac", "1", "-ar", str(config.SR), "-f", "f32le", "-"],
                       capture_output=True, timeout=600)
    a = np.frombuffer(r.stdout, np.float32)
    meldung = r.stderr.decode(errors="replace")[:200]
    if r.returncode and any(t in meldung for t in LESEFEHLER):
        raise Voruebergehend("Lesefehler: " + meldung)
    if a.size < config.SR * 5:
        raise RuntimeError("zu kurz oder nicht dekodierbar: " + meldung)
    # abgebrochen gelesen (z. B. NAS-Aussetzer mitten im Lied)? Dann nicht als fertig werten
    erwartet = min(dauer or 0, config.MAX_ANALYSE_S) * config.SR
    if r.returncode and a.size < 0.9 * erwartet:
        raise Voruebergehend(f"nur {a.size / config.SR:.0f} s von {erwartet / config.SR:.0f} s gelesen: " + meldung)
    return a.copy()


def _energiekurve(a, fenster_s=0.5):
    n = int(config.SR * fenster_s)
    k = a.size // n
    rms = np.sqrt((a[: k * n].reshape(k, n) ** 2).mean(1) + 1e-12)
    glatt = np.convolve(rms, np.ones(16) / 16, mode="same")  # ~8 s geglättet
    return glatt, fenster_s


def snippet_start(a, dauer):
    """Erster Abschnitt mit voller Besetzung; sonst 1:45; kurze Lieder 40 %."""
    dauer = dauer or a.size / config.SR
    if dauer < config.SNIPPET_S + 10:
        return 0.0, "anteil"
    kurve, schritt = _energiekurve(a)
    oben = np.percentile(kurve, 95)
    if oben > 1e-3:  # unter ~-60 dBFS ist da nichts zu finden (Stille, kaputte Datei)
        start_idx = int(15 / schritt)
        treffer = np.nonzero(kurve[start_idx:] >= 0.8 * oben)[0]
        if treffer.size:
            t = (treffer[0] + start_idx) * schritt
            # ein paar Sekunden vor den vollen Teil, damit der Einstieg hörbar ist
            t = max(0.0, t - 2)
            if t + config.SNIPPET_S <= dauer:
                return round(t, 1), "energie"
    if dauer >= config.SNIPPET_RUECKFALL + config.SNIPPET_S:
        return float(config.SNIPPET_RUECKFALL), "1:45"
    return round(dauer * 0.4, 1), "anteil"


def hauptteil_proben(a, fenster_s=30, anzahl=5, ab_anteil=0.6):
    """Bis zu `anzahl` × `fenster_s` Sekunden aus Abschnitten mit voller Besetzung.

    Voll = mittlere Energie im Fenster ≥ 60 % des 95. Perzentils. Kurze Lieder
    (≤ anzahl × fenster_s) bleiben ganz. Rückgabe: (Proben aneinander, ein
    zusammenhängendes Stück von bis zu 60 s für die Tempo-Schätzung).
    """
    n = int(fenster_s * config.SR)
    if a.size <= anzahl * n:
        return a, a[: 2 * n]
    kurve, schritt = _energiekurve(a)
    breite = int(fenster_s / schritt)
    mittel = np.convolve(kurve, np.ones(breite) / breite, mode="valid")  # Fenster ab Index i
    oben = np.percentile(kurve, 95)
    kandidaten = np.nonzero(mittel >= ab_anteil * oben)[0] if oben > 1e-3 else np.zeros(0, int)
    if kandidaten.size == 0:
        kandidaten = np.arange(mittel.size)
    wahl = kandidaten[np.linspace(0, kandidaten.size - 1, anzahl).round().astype(int)]
    starts = sorted({min(int(i * schritt * config.SR), a.size - n) for i in wahl})
    proben = np.concatenate([a[s:s + n] for s in starts])
    return proben, a[starts[0]: starts[0] + 2 * n]


def _fingerabdruck(a):
    import acoustid
    import chromaprint
    teil = a[: config.SR * config.FP_SEKUNDEN]
    pcm = (np.clip(teil, -1, 1) * 32767).astype("<i2").tobytes()
    kodiert = acoustid.fingerprint(config.SR, 1, iter([pcm]), maxlength=config.FP_SEKUNDEN)
    roh, _ = chromaprint.decode_fingerprint(kodiert)
    return np.asarray(roh, dtype=np.int64).astype(np.uint32).view(np.int32)


STUMM_DB = -60.0   # Pegel (RMS, dBFS) über das ganze Gehörte: darunter ist nichts zu hören (leise Klassik ~ -35)


def _pegel(a):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(a, dtype=np.float64))) + 1e-12)) if a.size else -240.0


def stumme_aussortieren(ids, ausgabe=print):
    """Erfasste Dateien nachprüfen: ist über die ganze Länge nichts zu hören → „stumm“ (nichts gelöscht)."""
    con = db.verbinden()
    if not nas_da():
        raise SystemExit(f"{config.MEDIA_ROOT} ist nicht eingebunden; nichts geändert")
    stumm = []
    for lid in ids:
        r = con.execute("SELECT pfad FROM lieder WHERE id=? AND status='analysiert'", (lid,)).fetchone()
        if r is None:
            continue
        pegel = _pegel(_dekodieren(str(config.medienpfad(r[0]))))
        ausgabe(f"{lid}: {pegel:.1f} dB {r[0]}")
        if pegel < STUMM_DB:
            stumm.append(lid)
    with con:
        con.executemany("UPDATE lieder SET status='stumm' WHERE id=?", [(i,) for i in stumm])
        n_ungueltig = gruppen_ungueltig_machen(con)
    ausgabe(f"{len(stumm)} von {len(ids)} stumm, Duplikatgruppen ungültig {n_ungueltig}")
    return stumm


def _analysiere(zeile):
    lied_id, pfad = zeile  # absoluter, schon geprüfter Pfad (der Hauptprozess kennt die Wurzel)
    t0 = time.time()
    try:
        info = _ffprobe(pfad)
        if info.pop("video"):
            return lied_id, {**info, "status": "video", "fehler": None, "analysiert": time.time(),
                             "analyse_s": round(time.time() - t0, 2)}
        a = _dekodieren(pfad, info["dauer"])
        if _pegel(a) < STUMM_DB:
            return lied_id, {**info, "status": "stumm", "fehler": None, "analysiert": time.time(),
                             "analyse_s": round(time.time() - t0, 2)}
        start, quelle = snippet_start(a, info["dauer"])
        fp = _fingerabdruck(a)
        # Das Klang-Modell kostet ~1,5 s je Minute Audio: es hört nur bis zu
        # 5 × 30 s aus dem vollen Hauptteil statt des ganzen Lieds (Mixe bis 15 min)
        proben, bpm_teil = hauptteil_proben(a)
        bpm = float(_modell["bpm"](bpm_teil)) or None
        kern = _modell["effnet"](proben)  # (Zeitfenster, 1280)
        mittel = kern.mean(0)
        p = _modell["kopf"](kern).mean(0)
        top = [(_modell["klassen"][i], round(float(p[i]), 3)) for i in p.argsort()[::-1][:5]]
        info.update(
            status="analysiert", fehler=None, bpm=bpm,
            snippet_start=start, snippet_quelle=quelle,
            emb=mittel.astype(np.float16).tobytes(), discogs=json.dumps(top, ensure_ascii=False),
            fp=fp.tobytes(), analysiert=time.time(), analyse_s=round(time.time() - t0, 2),
        )
    except (Voruebergehend, OSError, subprocess.TimeoutExpired) as e:
        # vielleicht nur ein NAS-Aussetzer: „neu“ lassen, der Lauf versucht es später nochmal
        info = {"status": "neu", "fehler": f"{type(e).__name__}: {e}"[:500]}
    except Exception as e:  # einzelne kaputte Dateien dürfen den Lauf nicht stoppen
        info = {"status": "fehler", "fehler": f"{type(e).__name__}: {e}"[:500],
                "analysiert": time.time(), "analyse_s": round(time.time() - t0, 2)}
    return lied_id, info


def analysieren(anzahl=None, zufall=False, jobs=2):
    """Alle „neuen“ Lieder analysieren. Jedes Ergebnis wird sofort in einer kurzen
    Transaktion geschrieben (die App bleibt bedienbar). Stürzt ein Arbeitsprozess ab,
    wird der Pool neu gestartet; ein Lied, das dreimal abstürzt, gilt als „fehler“.
    Fällt das NAS weg, bricht der Lauf ab, ohne Lieder als fehlerhaft zu markieren.
    """
    con = db.verbinden()
    with con:
        con.execute("UPDATE lieder SET status='fehler', fehler='Analyse stürzte wiederholt ab' "
                    "WHERE status='neu' AND versuche >= ?", (MAX_VERSUCHE,))
    zeilen = [(r["id"], r["pfad"]) for r in con.execute("SELECT id, pfad FROM lieder WHERE status='neu'")]
    if zufall:
        random.Random(42).shuffle(zeilen)
    # schon bewertete Lieder zuerst: ohne Steckbrief kann das Modell nicht von ihnen lernen
    bewertet = {r[0] for r in con.execute("SELECT DISTINCT lied_id FROM etiketten")}
    zeilen.sort(key=lambda z: z[0] not in bewertet)
    if anzahl:
        zeilen = zeilen[:anzahl]
    if not zeilen:
        print("nichts zu tun")
        return 0
    if not nas_da():
        raise SystemExit(f"{config.MEDIA_ROOT} ist nicht eingebunden")
    print(f"analysiere {len(zeilen)} Lieder mit {jobs} Prozessen", flush=True)
    t0 = time.time()
    fertig = fehler = voruebergehend = 0
    offen = iter(zeilen)
    kontext = multiprocessing.get_context("forkserver")

    def neuer_pool():
        return ProcessPoolExecutor(jobs, mp_context=kontext, initializer=_init_worker, max_tasks_per_child=500)

    pool, laufend, kaputt = neuer_pool(), {}, False
    try:
        while True:
            while len(laufend) < 2 * jobs:
                zeile = next(offen, None)
                if zeile is None:
                    break
                with con:  # vorher zählen: ein Absturz soll nicht ewig wiederkommen
                    con.execute("UPDATE lieder SET versuche=versuche+1 WHERE id=?", (zeile[0],))
                try:
                    auftrag = (zeile[0], str(config.medienpfad(zeile[1])))
                except ValueError as e:
                    with con:
                        con.execute("UPDATE lieder SET status='fehler', fehler=? WHERE id=?", (str(e), zeile[0]))
                    continue
                laufend[pool.submit(_analysiere, auftrag)] = zeile[0]
            if not laufend:
                break
            erledigt, _ = wait(laufend, return_when=FIRST_COMPLETED)
            for f in erledigt:
                laufend.pop(f)
                try:
                    lied_id, info = f.result()
                except BrokenProcessPool:
                    kaputt = True
                    continue  # Lied bleibt „neu“, versuche ist schon hochgezählt
                if info["status"] != "analysiert" and not nas_da():
                    # NAS weg: dieses Lied war es nicht, sondern das Netz
                    with con:
                        con.execute("UPDATE lieder SET status='neu', versuche=max(versuche-1, 0) WHERE id=?", (lied_id,))
                    raise SystemExit("NAS weg: Abbruch, Lieder bleiben „neu“")
                if info["status"] == "neu":
                    voruebergehend += 1
                if info["status"] == "analysiert":
                    info["versuche"] = 0
                spalten = ", ".join(f"{k}=?" for k in info)
                with con:
                    con.execute(f"UPDATE lieder SET {spalten} WHERE id=?", (*info.values(), lied_id))
                fertig += 1
                fehler += info["status"] == "fehler"
                if fertig % 20 == 0:
                    rate = fertig / (time.time() - t0)
                    rest = (len(zeilen) - fertig) / rate if rate else 0
                    print(f"{fertig}/{len(zeilen)}  {rate * 3600:.0f}/h  Fehler {fehler}  "
                          f"Lesefehler {voruebergehend}  Rest ~{rest / 3600:.1f} h", flush=True)
            if kaputt:
                kaputt = False
                print("Arbeitsprozess abgestürzt: Pool neu gestartet", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                laufend.clear()
                pool = neuer_pool()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    print(f"fertig: {fertig}, Fehler {fehler}, Lesefehler {voruebergehend}", flush=True)
    return fertig


def videos_aussortieren(ausgabe=print):
    """Schon erfasste Dateien in Containern, die Video tragen können, nachprüfen: mit Videospur → „video“.
    Bewertungen bleiben stehen (nichts gelöscht); App, Training und Duplikate sehen nur „analysiert“."""
    con = db.verbinden()
    if not nas_da():
        raise SystemExit(f"{config.MEDIA_ROOT} ist nicht eingebunden; nichts geändert")
    zeilen = con.execute("SELECT id, pfad FROM lieder WHERE status IN ('analysiert','fehler','neu') "
                         "AND (lower(pfad) LIKE '%.mp4' OR lower(pfad) LIKE '%.m4a')").fetchall()
    videos = []
    for lid, rel in zeilen:
        try:
            if _ffprobe(str(config.medienpfad(rel)))["video"]:
                videos.append(lid)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as e:
            ausgabe(f"übersprungen: {rel}: {e}")
    with con:
        con.executemany("UPDATE lieder SET status='video' WHERE id=?", [(i,) for i in videos])
        n_ungueltig = gruppen_ungueltig_machen(con)
    ausgabe(f"{len(zeilen)} geprüft, {len(videos)} Videos aussortiert, Duplikatgruppen ungültig {n_ungueltig}")
    return videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("schritt", choices=["suchen", "analysieren", "videos", "stumm"])
    ap.add_argument("ids", nargs="*", type=int, help="bei „stumm“: diese Lied-IDs nachprüfen")
    ap.add_argument("-n", "--anzahl", type=int)
    ap.add_argument("--zufall", action="store_true")
    ap.add_argument("-j", "--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.schritt == "suchen":
        suchen()
    elif a.schritt == "videos":
        videos_aussortieren()
    elif a.schritt == "stumm":
        stumme_aussortieren(a.ids)
    elif analysieren(a.anzahl, a.zufall, a.jobs):
        # nur wenn etwas dazukam: Duplikate neu berechnen (sonst stündlich unnötig)
        from . import duplikate
        duplikate.finden()


if __name__ == "__main__":
    main()
