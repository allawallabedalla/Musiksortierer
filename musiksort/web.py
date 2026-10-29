"""Web-App (FastAPI): Bewerten am iPhone, Genres pflegen, Duplikate entscheiden, Stand.

  uvicorn musiksort.web:app --host 0.0.0.0 --port 8097

Endpunkte und ihr Verhalten: siehe tests/test_web.py.
"""
import contextvars
import hmac
import json
import os
import subprocess
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from . import config, db, dupauto, duplikate, genres as genrelib, klang, lernen, sprache

STATIC = Path(__file__).resolve().parent / "static"
SCHLUESSEL = os.environ.get("MUSIKSORT_SCHLUESSEL", "")  # leer = kein Zugangsschlüssel
MAX_KOERPER = 64 * 1024
ANALYSE_CGROUP = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/"
                      "app.slice/musiksort-analyse.service")
# Direkt abspielbar (ganze Datei mit Scrub-Leiste); alles andere bekommt den erzeugten Ausschnitt
DIREKT = {"mp3": "audio/mpeg", "aac": "audio/mp4", "alac": "audio/mp4", "flac": "audio/flac",
          "pcm_s16le": "audio/wav", "pcm_s24le": "audio/wav", "pcm_s32le": "audio/wav", "pcm_f32le": "audio/wav"}

app = FastAPI(title="Musiksortierer", docs_url=None, redoc_url=None, openapi_url=None)
_verbindungen = contextvars.ContextVar("verbindungen", default=None)

hosts = [h.strip() for h in os.environ.get("MUSIKSORT_HOSTS", "").split(",") if h.strip()]
if hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)


@app.middleware("http")
async def _aufraeumen(request: Request, call_next):
    """Jede Verbindung einer Anfrage wird danach geschlossen (auch nach Fehlern: keine hängende Schreibsperre)."""
    liste = []
    _verbindungen.set(liste)
    try:
        return await call_next(request)
    finally:
        for c in liste:
            try:
                c.rollback()
                c.close()
            except Exception:
                pass


@app.middleware("http")
async def _schutz(request: Request, call_next):
    laenge = request.headers.get("content-length")
    if laenge and laenge.isdigit() and int(laenge) > MAX_KOERPER:
        return JSONResponse({"detail": "Anfrage zu groß"}, status_code=413)
    if SCHLUESSEL:
        frage = request.query_params.get("schluessel")
        if request.url.path == "/" and frage is not None:
            if hmac.compare_digest(frage, SCHLUESSEL):
                r = RedirectResponse("/", status_code=303)
                r.set_cookie("schluessel", SCHLUESSEL, max_age=400 * 86400, httponly=True, samesite="strict")
                return r
            return JSONResponse({"detail": "Schlüssel falsch"}, status_code=401)
        if not hmac.compare_digest(request.cookies.get("schluessel", ""), SCHLUESSEL):
            return JSONResponse({"detail": "Zugang nur mit Schlüssel"}, status_code=401)
    r = await call_next(request)
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["Referrer-Policy"] = "no-referrer"
    return r


@app.exception_handler(OverflowError)
async def _zu_gross(request, exc):
    return JSONResponse({"detail": "Zahl zu groß"}, status_code=400)


@app.exception_handler(Exception)
async def _fehler(request, exc):
    print("Fehler:", request.url.path, repr(exc), flush=True)
    return JSONResponse({"detail": f"Serverfehler: {type(exc).__name__}"}, status_code=500)


def con():
    c = db.verbinden(schema=False, threads=True)
    liste = _verbindungen.get()
    if liste is not None:
        liste.append(c)
    return c


def _wer(request: Request):
    return (request.headers.get("user-agent") or "")[:80]


@app.on_event("startup")
def _start():
    c = db.verbinden()  # Schema anlegen/nachrüsten
    c.close()
    def laden():
        c = db.verbinden(schema=False, threads=True)
        try:
            lernen.bestand.aktualisieren(c)
        finally:
            c.close()
    threading.Thread(target=laden, daemon=True).start()


# ---- Genres

def _genre_pfade(c, zeilen=None):
    zeilen = zeilen if zeilen is not None else c.execute("SELECT id, name, eltern_id FROM genres").fetchall()
    eltern = {r[0]: (r[1], r[2]) for r in zeilen}
    pfade = {}
    for gid in eltern:
        teile, e, gesehen = [], gid, set()
        while e is not None and e in eltern and e not in gesehen:  # Kreis-Schutz
            gesehen.add(e)
            teile.append(eltern[e][0])
            e = eltern[e][1]
        pfade[gid] = " › ".join(reversed(teile))
    return pfade


def _genres(c):
    zeilen = c.execute("SELECT id, name, eltern_id FROM genres").fetchall()
    pfade = _genre_pfade(c, zeilen)
    n = dict(c.execute("SELECT genre_id, COUNT(*) FROM etiketten WHERE haupt=1 GROUP BY genre_id").fetchall())
    return [{"id": r[0], "name": r[1], "eltern_id": r[2], "pfad": pfade[r[0]], "n": n.get(r[0], 0)}
            for r in sorted(zeilen, key=lambda r: pfade[r[0]].lower())]


@app.get("/api/genres")
def api_genres():
    c = con()
    c.execute("BEGIN")  # Momentaufnahme: Namen, Pfade und Zahlen aus einem Stand
    return _genres(c)


@app.get("/api/namen")
def api_namen(q: str = Query("", max_length=60)):
    c = con()
    return genrelib.namen_vorschlagen(c, q, _genre_pfade(c))


class GenreNeu(BaseModel):
    name: str = Field(max_length=60)
    eltern_id: Optional[int] = None
    eltern_name: Optional[str] = Field(None, max_length=60)


def _name_pruefen(name):
    name = " ".join((name or "").split())
    if not name:
        raise HTTPException(400, "Name fehlt")
    if "/" in name:
        raise HTTPException(400, "Bitte ohne „/“ (wird später ein Ordnername)")
    return name


def _anlegen(c, name, eltern_id):
    for r in c.execute("SELECT id, name FROM genres WHERE eltern_id IS ?", (eltern_id,)):
        if genrelib.norm(r[1]) == genrelib.norm(name):
            return r[0]
    return c.execute("INSERT INTO genres(name, eltern_id, erstellt) VALUES (?,?,?)",
                     (name, eltern_id, db.jetzt())).lastrowid


@app.post("/api/genres")
def api_genre_neu(k: GenreNeu):
    name = _name_pruefen(k.name)
    if k.eltern_name is not None and k.eltern_id is not None:
        raise HTTPException(400, "Entweder bestehendes oder neues Obergenre")
    ober = _name_pruefen(k.eltern_name) if k.eltern_name is not None else None
    c = con()
    with lernen.sperre():
        c.execute("BEGIN IMMEDIATE")
        eltern = k.eltern_id
        if eltern is not None and not c.execute("SELECT 1 FROM genres WHERE id=?", (eltern,)).fetchone():
            raise HTTPException(400, "Obergenre gibt es nicht")
        if ober is not None:
            eltern = _anlegen(c, ober, None)
        gid = _anlegen(c, name, eltern)
        c.commit()
    return {"id": gid}


class GenreAendern(BaseModel):
    name: str = Field(max_length=60)
    eltern_id: Optional[int] = None


@app.patch("/api/genres/{gid}")
def api_genre_aendern(gid: int, k: GenreAendern):
    c = con()
    with lernen.sperre():
        c.execute("BEGIN IMMEDIATE")
        if not c.execute("SELECT 1 FROM genres WHERE id=?", (gid,)).fetchone():
            raise HTTPException(404, "Genre gibt es nicht (mehr)")
        if k.eltern_id is not None and not c.execute("SELECT 1 FROM genres WHERE id=?", (k.eltern_id,)).fetchone():
            raise HTTPException(400, "Obergenre gibt es nicht")
        try:
            genrelib.aendern(c, gid, k.name, k.eltern_id, _commit=False)
        except genrelib.GenreFehler as e:
            c.rollback()
            raise HTTPException(409 if "Zusammenführen" in str(e) else 400, str(e))
        c.commit()
    return {"ok": True}


class Zusammen(BaseModel):
    ziel_id: int


@app.post("/api/genres/{gid}/zusammenfuehren")
def api_zusammenfuehren(gid: int, k: Zusammen):
    c = con()
    with lernen.sperre():
        c.execute("BEGIN IMMEDIATE")
        da = {r[0] for r in c.execute("SELECT id FROM genres WHERE id IN (?,?)", (gid, k.ziel_id))}
        if da != {gid, k.ziel_id}:
            raise HTTPException(404, "Genre gibt es nicht (mehr)")
        try:
            genrelib.zusammenfuehren(c, gid, k.ziel_id, _commit=False)
        except genrelib.GenreFehler as e:
            c.rollback()
            raise HTTPException(400, str(e))
        c.commit()
    lernen.trainieren_im_hintergrund()
    return {"ok": True}


@app.get("/api/verwechslungen")
def api_verwechslungen():
    c = con()
    pfade = _genre_pfade(c)
    return [{"a": r[0], "b": r[1], "anteil": r[2], "n": r[3], "a_pfad": pfade[r[0]], "b_pfad": pfade[r[1]]}
            for r in c.execute("SELECT a, b, anteil, n FROM verwechslungen ORDER BY anteil DESC")
            if r[0] in pfade and r[1] in pfade]


# ---- Bewerten

def _lied(c, lid):
    r = c.execute("SELECT * FROM lieder WHERE id=?", (lid,)).fetchone()
    if not r:
        return None
    return {
        "id": r["id"], "pfad": r["pfad"], "dateiname": r["pfad"].rsplit("/", 1)[-1],
        "interpret": r["interpret"], "titel": r["titel"], "album": r["album"], "tag_genre": r["tag_genre"],
        "format": (r["codec"] or "").upper() or None, "codec": r["codec"], "bitrate": r["bitrate"],
        "bpm": round(r["bpm"]) if r["bpm"] else None, "dauer": r["dauer"],
        "snippet_start": r["snippet_start"] or 0, "snippet_quelle": r["snippet_quelle"] or "–",
        "discogs": json.loads(r["discogs"]) if r["discogs"] else [],
        "direkt": (r["codec"] or "") in DIREKT,
        "stern": bool(r["stern"]),
        "sprache": sprache.anzeige(r["sprache"], r["sprache_p"]),
    }


@app.get("/api/naechste")
def api_naechste(anzahl: int = Query(3, ge=1, le=10), ohne: str = Query("", max_length=200)):
    c = con()
    aus = [int(x) for x in ohne.split(",") if x.strip().isdigit()]
    pfade = _genre_pfade(c)
    erg = []
    for n in lernen.naechste(c, anzahl, aus):
        v = c.execute("SELECT genre_id, p, zweit_id, p2 FROM vorhersagen WHERE lied_id=?", (n["lied_id"],)).fetchone()
        vorschlag = []
        if v:
            for gid, p in ((v[0], v[1]), (v[2], v[3])):
                if gid in pfade and p is not None:
                    vorschlag.append({"id": gid, "pfad": pfade[gid], "p": round(p, 3)})
        lied = _lied(c, n["lied_id"])
        if lied:
            erg.append({**n, "lied": lied, "vorschlag": vorschlag,
                        "vorhergesagt": n.get("pruef_id") or (vorschlag[0]["id"] if vorschlag else None)})
    return erg


class Bewertung(BaseModel):
    lied_id: int
    haupt: int
    tags: list[int] = Field(default_factory=list, max_length=20)
    stichprobe: bool = False
    vorschlag_id: Optional[int] = None


@app.post("/api/bewerten")
def api_bewerten(b: Bewertung):
    c = con()
    with lernen.sperre():
        c.execute("BEGIN IMMEDIATE")
        if not c.execute("SELECT 1 FROM lieder WHERE id=?", (b.lied_id,)).fetchone():
            raise HTTPException(404, "Lied unbekannt")
        alle = {b.haupt, *b.tags}
        da = {r[0] for r in c.execute(f"SELECT id FROM genres WHERE id IN ({','.join('?' * len(alle))})", list(alle))}
        if da != alle:
            raise HTTPException(409, "Ein Genre gibt es nicht mehr. Bitte neu wählen.")
        jetzt = db.jetzt()
        quelle = "stichprobe" if b.stichprobe else "ich"
        vorher = _vorher(c, b.lied_id)
        c.execute("DELETE FROM etiketten WHERE lied_id=?", (b.lied_id,))
        c.execute("INSERT INTO etiketten VALUES (?,?,1,?,?)", (b.lied_id, b.haupt, quelle, jetzt))
        c.executemany("INSERT OR IGNORE INTO etiketten VALUES (?,?,0,?,?)",
                      [(b.lied_id, t, quelle, jetzt) for t in b.tags if t != b.haupt])
        # Stichprobe zählt für das Genre, das das Modell beim Laden vorhergesagt hat; gibt es das nicht mehr: verwerfen
        sp_id = pruefung = None
        if b.stichprobe and b.vorschlag_id is not None and \
                c.execute("SELECT 1 FROM genres WHERE id=?", (b.vorschlag_id,)).fetchone():
            sp_id = c.execute("INSERT INTO stichproben(lied_id, vorhergesagt, gewaehlt, ts) VALUES (?,?,?,?)",
                              (b.lied_id, b.vorschlag_id, b.haupt, jetzt)).lastrowid
            # Rückmeldung erst nach der Wahl (vorher würde sie die Prüfung schönen); richtig wie in lernen._stichproben
            teil = lernen._baum(c)[0]
            pruefung = {"richtig": b.haupt in teil.get(b.vorschlag_id, {b.vorschlag_id}),
                        "modell": _genre_pfade(c).get(b.vorschlag_id)}
        c.execute("DELETE FROM uebersprungen WHERE lied_id=?", (b.lied_id,))
        rid = _merken(c, b.lied_id, "bewertet", vorher, sp_id)
        c.commit()
    lernen.trainieren_im_hintergrund()
    return {"ok": True, "rueck_id": rid, "stichprobe": pruefung}


def _vorher(c, lid):
    return {"etiketten": [list(r) for r in c.execute("SELECT genre_id, haupt, quelle, ts FROM etiketten WHERE lied_id=?", (lid,))],
            "uebersprungen": (list(r) if (r := c.execute("SELECT ts, anzahl FROM uebersprungen WHERE lied_id=?",
                                                          (lid,)).fetchone()) else None)}


def _merken(c, lid, art, vorher, sp_id=None):
    # nur die jüngsten Schritte aufheben: der Knopf geht höchstens ein paar Lieder zurück
    c.execute("DELETE FROM rueckgaengig WHERE id <= (SELECT max(id) - 200 FROM rueckgaengig)")
    return c.execute("INSERT INTO rueckgaengig(lied_id, art, vorher, stichprobe_id, ts) VALUES (?,?,?,?,?)",
                     (lid, art, json.dumps(vorher), sp_id, db.jetzt())).lastrowid


class Rueckgaengig(BaseModel):
    rueck_id: int


@app.post("/api/rueckgaengig")
def api_rueckgaengig(r: Rueckgaengig):
    """Letzte Bewertung bzw. „Später“ zurücknehmen: Etiketten, Stichprobe und Überspringen wie vorher."""
    c = con()
    with lernen.sperre():
        c.execute("BEGIN IMMEDIATE")
        z = c.execute("SELECT lied_id, art, vorher, stichprobe_id FROM rueckgaengig WHERE id=?", (r.rueck_id,)).fetchone()
        if not z:
            c.rollback()
            raise HTTPException(404, "Lässt sich nicht mehr zurücknehmen")
        lid, vorher = z["lied_id"], json.loads(z["vorher"])
        da = {g[0] for g in c.execute("SELECT id FROM genres")}
        c.execute("DELETE FROM etiketten WHERE lied_id=?", (lid,))
        c.executemany("INSERT INTO etiketten VALUES (?,?,?,?,?)",
                      [(lid, g, h, q, t) for g, h, q, t in vorher["etiketten"] if g in da])
        if z["stichprobe_id"] is not None:
            c.execute("DELETE FROM stichproben WHERE id=?", (z["stichprobe_id"],))
        c.execute("DELETE FROM uebersprungen WHERE lied_id=?", (lid,))
        if vorher["uebersprungen"]:
            c.execute("INSERT INTO uebersprungen(lied_id, ts, anzahl) VALUES (?,?,?)", (lid, *vorher["uebersprungen"]))
        c.execute("DELETE FROM rueckgaengig WHERE id=?", (r.rueck_id,))
        c.commit()
    if z["art"] == "bewertet":
        lernen.trainieren_im_hintergrund()
    return {"ok": True, "lied_id": lid}


class Stern(BaseModel):
    lied_id: int
    an: bool


@app.post("/api/stern")
def api_stern(s: Stern, request: Request):
    """Favorit an/aus. In die Datei geschrieben wird später (tags.py), sobald die Freigabe beschreibbar ist."""
    c = con()
    c.execute("BEGIN IMMEDIATE")
    if not c.execute("SELECT 1 FROM lieder WHERE id=?", (s.lied_id,)).fetchone():
        raise HTTPException(404, "Lied unbekannt")
    c.execute("UPDATE lieder SET stern=?, stern_ts=? WHERE id=?", (int(s.an), db.jetzt(), s.lied_id))
    duplikate.protokoll(c, "stern_an" if s.an else "stern_aus", {"lied": s.lied_id}, _wer(request))
    c.commit()
    return {"ok": True, "stern": s.an}


class Ueberspringen(BaseModel):
    lied_id: int


@app.post("/api/ueberspringen")
def api_ueberspringen(u: Ueberspringen):
    c = con()
    c.execute("BEGIN IMMEDIATE")
    if not c.execute("SELECT 1 FROM lieder WHERE id=?", (u.lied_id,)).fetchone():
        raise HTTPException(404, "Lied unbekannt")
    vorher = _vorher(c, u.lied_id)
    c.execute("INSERT INTO uebersprungen(lied_id, ts) VALUES (?,?) ON CONFLICT(lied_id) "
              "DO UPDATE SET ts=excluded.ts, anzahl=anzahl+1", (u.lied_id, db.jetzt()))
    rid = _merken(c, u.lied_id, "uebersprungen", vorher)
    c.commit()
    return {"ok": True, "rueck_id": rid}


@app.get("/api/aehnliche/{lid}")
def api_aehnliche(lid: int, n: int = Query(20, ge=1, le=50)):
    c = con()
    if not c.execute("SELECT 1 FROM lieder WHERE id=? AND status='analysiert' AND emb IS NOT NULL", (lid,)).fetchone():
        raise HTTPException(404, "Lied noch nicht analysiert")
    lernen.bestand.aktualisieren(c, brauche=[lid])
    erg = []
    for i, s in lernen.bestand.aehnliche(lid, n):
        r = c.execute("SELECT pfad, interpret, titel FROM lieder WHERE id=?", (i,)).fetchone()
        if r:
            erg.append({"id": i, "interpret": r[1], "titel": r[2], "dateiname": r[0].rsplit("/", 1)[-1],
                        "aehnlichkeit": round(s, 3)})
    return erg


# ---- Audio

_datei_sperren = OrderedDict()
_sperren_sperre = threading.Lock()


def _sperre_fuer(schluessel):
    with _sperren_sperre:
        s = _datei_sperren.pop(schluessel, None) or threading.Lock()
        _datei_sperren[schluessel] = s
        while len(_datei_sperren) > 500:
            _datei_sperren.popitem(last=False)
        return s


def _quelle(c, lid):
    r = c.execute("SELECT pfad, dauer, snippet_start, codec FROM lieder WHERE id=?", (lid,)).fetchone()
    if not r:
        raise HTTPException(404, "Lied unbekannt")
    try:
        pfad = config.medienpfad(r[0])
    except ValueError:
        raise HTTPException(400, "Pfad außerhalb der Freigabe")
    if not pfad.is_file():
        raise HTTPException(404, "Datei nicht (mehr) da")
    return pfad, r[1] or 0, r[2] or 0, r[3]


def cache_aufraeumen(max_bytes=None):
    """Ausschnitt-Cache klein halten: über der Grenze die ältesten löschen, bis 80 % erreicht sind."""
    max_bytes = max_bytes or config.SNIPPET_CACHE_MB * 1024 * 1024
    try:
        dateien = [(f.stat().st_mtime, f.stat().st_size, f) for f in config.SNIPPETS.glob("*.mp3")
                   if not f.name.endswith(".tmp.mp3")]
    except OSError:
        return
    summe = sum(s for _, s, _ in dateien)
    if summe <= max_bytes:
        return
    for _, s, f in sorted(dateien, key=lambda x: x[0]):
        if summe <= 0.8 * max_bytes:
            break
        f.unlink(missing_ok=True)
        summe -= s


_letztes_aufraeumen = [0.0]


@app.get("/api/snippet/{lid}")
def api_snippet(lid: int, request: Request, ausschnitt: int = Query(0, ge=0, le=5)):
    c = con()
    pfad, dauer, start, _ = _quelle(c, lid)
    c.close()
    if ausschnitt:  # weitere Stellen gleichmäßig übers Lied verteilt
        start = max(0.0, (dauer - config.SNIPPET_S) * ausschnitt / 6)
    config.SNIPPETS.mkdir(parents=True, exist_ok=True)
    ziel = config.SNIPPETS / f"{lid}_{ausschnitt}.mp3"
    with _sperre_fuer(ziel.name):
        if not ziel.exists():
            tmp = config.SNIPPETS / f"{lid}_{ausschnitt}_{uuid.uuid4().hex}.tmp.mp3"
            try:
                r = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.1f}", "-i", str(pfad),
                                    "-t", str(config.SNIPPET_S), "-vn", "-ac", "2", "-b:a", "160k",
                                    "-af", "afade=t=in:d=0.5", "-f", "mp3", str(tmp)],
                                   capture_output=True, timeout=120)
                if r.returncode or not tmp.exists() or tmp.stat().st_size == 0:
                    raise HTTPException(422, "Ausschnitt ließ sich nicht erzeugen")
                tmp.replace(ziel)
            finally:
                tmp.unlink(missing_ok=True)
    os.utime(ziel)  # LRU: zuletzt gehört bleibt
    if time.time() - _letztes_aufraeumen[0] > 300:
        _letztes_aufraeumen[0] = time.time()
        threading.Thread(target=cache_aufraeumen, daemon=True).start()
    return FileResponse(ziel, media_type="audio/mpeg",
                        headers={"X-Start": f"{start:.1f}", "Cache-Control": "private, max-age=86400"})


@app.get("/api/datei/{lid}")
def api_datei(lid: int):
    c = con()
    r = c.execute("SELECT codec FROM lieder WHERE id=?", (lid,)).fetchone()
    if r and (r[0] or "") not in DIREKT:  # vor dem Dateizugriff: WMA & Co. nie ausliefern
        raise HTTPException(415, "Format nicht direkt abspielbar")
    pfad, _, _, codec = _quelle(c, lid)
    return FileResponse(pfad, media_type=DIREKT[codec], headers={"Cache-Control": "private, max-age=3600"})


class GeraeteProtokoll(BaseModel):
    model_config = ConfigDict(extra="allow")
    ereignis: str = Field(max_length=40)


@app.post("/api/protokoll")
def api_protokoll(p: GeraeteProtokoll, request: Request):
    daten = json.dumps(p.model_dump(), ensure_ascii=False)[:2000]
    print("Gerät:", daten, "|", _wer(request), flush=True)
    return {"ok": True}


# ---- Stand und Fortschritt

def _bewertungen_mind():
    n = config.MIN_BEWERTUNGEN_AUTO
    while lernen.wilson_unten(n, n) < config.ZIEL_TREFFERQUOTE:
        n += 1
    return n


def _stichproben_noetig(ok, n):
    """Wie viele Stichproben insgesamt, wenn ab jetzt alle stimmen (vorsichtige Untergrenze ≥ STICHPROBE_ZIEL)."""
    m = max(n, config.STICHPROBE_MIN)
    while lernen.wilson_unten(ok + (m - n), m) < config.STICHPROBE_ZIEL:
        m += 1
        if m > 10000:
            return None
    return m


@app.get("/api/stand")
def api_stand():
    c = con()
    c.execute("BEGIN")
    pfade = _genre_pfade(c)
    modell = []
    for r in c.execute("SELECT * FROM modell_stand"):
        if r["genre_id"] not in pfade:
            continue
        modell.append({**dict(r), "pfad": pfade[r["genre_id"]],
                       "stichprobe_noetig": _stichproben_noetig(r["stichprobe_ok"] or 0, r["stichprobe_n"] or 0)})
    modell.sort(key=lambda m: -(m["n"] or 0))
    gruppen = c.execute(f"SELECT COUNT(*) FROM dup_gruppen g WHERE {duplikate.OFFENE_GRUPPEN}").fetchone()[0]
    ueber = c.execute(f"SELECT COUNT(*) - COUNT(DISTINCT m.gruppe_id) FROM dup_mitglieder m JOIN dup_gruppen g "
                      f"ON g.id=m.gruppe_id WHERE {duplikate.OFFENE_GRUPPEN}").fetchone()[0]
    return {
        "modell": modell,
        "bewertet": c.execute("SELECT COUNT(DISTINCT lied_id) FROM etiketten").fetchone()[0],
        "automatisch": c.execute("SELECT COUNT(*) FROM vorhersagen WHERE auto=1").fetchone()[0],
        "min_auto": config.MIN_BEWERTUNGEN_AUTO, "ziel": config.ZIEL_TREFFERQUOTE,
        "stichprobe_min": config.STICHPROBE_MIN, "stichprobe_ziel": config.STICHPROBE_ZIEL,
        "bewertungen_mind": _bewertungen_mind(), "stichprobe_noetig": _stichproben_noetig(0, 0),
        "dup_gruppen": gruppen, "dup_ueberzaehlig": ueber or 0,
    }


def _tempo(c, stunden):
    """Analyse-Tempo je Stunde aus 15-min-Fenstern (Median, Spanne), dazu Verlauf je Stunde."""
    jetzt = db.jetzt()
    zeiten = [r[0] for r in c.execute("SELECT analysiert FROM lieder WHERE analysiert > ?", (jetzt - stunden * 3600,))]
    verlauf = [0] * stunden
    fenster = {}
    for t in zeiten:
        alter = jetzt - t
        verlauf[min(stunden - 1, int(alter // 3600))] += 1
        fenster[int(alter // 900)] = fenster.get(int(alter // 900), 0) + 1
    raten = sorted(n * 4 for n in fenster.values())
    tempo = raten[len(raten) // 2] if raten else 0
    return {
        "verlauf": [{"vor_stunden": i, "n": n} for i, n in enumerate(verlauf)],
        "letzte_stunde": verlauf[0], "tempo": tempo, "tempo_n": len(raten),
        "tempo_min": raten[0] if raten else 0, "tempo_max": raten[-1] if raten else 0,
    }


def _cg(datei):
    try:
        return (ANALYSE_CGROUP / datei).read_text().strip()
    except OSError:
        return None


def _dienst():
    procs = _cg("cgroup.procs")
    if not procs:
        return {"laeuft": False}
    mb = lambda w: None if w in (None, "max") else round(int(w) / 1048576)
    stat = dict(z.split() for z in (_cg("cpu.stat") or "").splitlines() if len(z.split()) == 2)
    perioden = int(stat.get("nr_periods", 0))
    return {"laeuft": True, "speicher_mb": mb(_cg("memory.current")), "bremse_mb": mb(_cg("memory.high")),
            "grenze_mb": mb(_cg("memory.max")),
            "cpu_gebremst": round(int(stat.get("nr_throttled", 0)) / perioden, 3) if perioden else 0}


@app.get("/api/analyse")
def api_analyse(stunden: int = Query(6, ge=2, le=72)):
    c = con()
    lieder = dict(c.execute("SELECT status, COUNT(*) FROM lieder GROUP BY status").fetchall())
    dauer = c.execute("SELECT avg(analyse_s) FROM lieder WHERE analysiert > ?", (db.jetzt() - 86400,)).fetchone()[0]
    return {"lieder": lieder, **_tempo(c, stunden), "dauer_s": round(dauer, 1) if dauer else None, "dienst": _dienst()}


@app.get("/api/fortschritt")
def api_fortschritt():
    c = con()
    c.execute("BEGIN")
    st = dict(c.execute("SELECT status, COUNT(*) FROM lieder WHERE status IN ('neu','analysiert','fehler') "
                        "GROUP BY status").fetchall())
    fertig = st.get("analysiert", 0) + st.get("fehler", 0)
    gesamt = fertig + st.get("neu", 0)
    t = _tempo(c, 6)
    rest = st.get("neu", 0)
    pfade = _genre_pfade(c)
    bewertet = c.execute("SELECT COUNT(DISTINCT e.lied_id) FROM etiketten e JOIN lieder l ON l.id=e.lied_id "
                         "WHERE l.status='analysiert'").fetchone()[0]
    # fein (auto) oder grob (Obergenre); zählt erst, wenn das jeweilige Genre durch Stichproben belegt ist
    stufe = ("SELECT COUNT(*) FROM vorhersagen v JOIN lieder l ON l.id=v.lied_id AND l.status='analysiert' "
             "LEFT JOIN modell_stand f ON f.genre_id=v.genre_id AND v.auto=1 "
             "LEFT JOIN modell_stand o ON o.genre_id=v.ober_id "
             "WHERE (v.auto=1 OR v.ober_id IS NOT NULL) AND v.lied_id NOT IN (SELECT lied_id FROM etiketten) AND ")
    bestaetigt = c.execute(stufe + "coalesce(f.belegt, o.belegt, 0)=1").fetchone()[0]
    wartet = c.execute(stufe + "coalesce(f.belegt, o.belegt, 0)=0").fetchone()[0]
    selbst = c.execute("SELECT COUNT(*) FROM modell_stand WHERE belegt=1 "
                       "AND (schwelle IS NOT NULL OR ober_schwelle IS NOT NULL)").fetchone()[0]
    naechstes = None
    for gid, n in c.execute("SELECT genre_id, COUNT(*) n FROM etiketten WHERE haupt=1 GROUP BY genre_id ORDER BY n DESC"):
        if n < config.MIN_BEWERTUNGEN_AUTO and gid in pfade:
            naechstes = {"id": gid, "pfad": pfade[gid], "fehlt": config.MIN_BEWERTUNGEN_AUTO - n}
            break
    offen = c.execute(f"SELECT COUNT(*) FROM dup_gruppen g WHERE {duplikate.OFFENE_GRUPPEN}").fetchone()[0]
    entschieden = c.execute("SELECT COUNT(*) FROM dup_gruppen WHERE status IN ('freigegeben','verworfen','erledigt')").fetchone()[0]
    return {
        "analyse": {"gesamt": gesamt, "fertig": fertig, "rest": rest, "pro_stunde": t["tempo"] or None,
                    "rest_stunden": rest / t["tempo"] if t["tempo"] and rest else (0 if not rest else None)},
        "genres": {"gesamt": gesamt, "bewertet": bewertet, "bestaetigt": bestaetigt,
                   "abgedeckt": bewertet + bestaetigt, "selbst_sortierend": selbst, "wartet": wartet,
                   "naechstes": naechstes},
        "duplikate": {"offen": offen, "entschieden": entschieden},
    }


# ---- Duplikate

@app.get("/api/duplikate")
def api_duplikate(grenze: int = Query(50, ge=1, le=200)):
    c = con()
    c.execute("BEGIN")
    erg = []
    for g in c.execute(f"SELECT g.id, g.sicher FROM dup_gruppen g WHERE {duplikate.OFFENE_GRUPPEN} "
                       "ORDER BY g.sicher DESC, g.id LIMIT ?", (grenze,)).fetchall():
        ms = c.execute("SELECT l.id, l.pfad, l.codec, l.bitrate, l.verlustfrei, l.bittiefe, l.samplerate, l.dauer, "
                       "l.interpret, l.titel, l.album, l.grenzfrequenz, l.kante, l.stern, m.behalten, m.abweichung FROM dup_mitglieder m "
                       "JOIN lieder l ON l.id=m.lied_id WHERE m.gruppe_id=?", (g[0],)).fetchall()
        if not ms:
            continue
        best = max(ms, key=duplikate.qualitaet)["id"]
        mitglieder = [{k: r[k] for k in ("id", "pfad", "codec", "bitrate", "verlustfrei", "bittiefe", "samplerate",
                                          "dauer", "behalten", "abweichung", "stern")}
                      | {"empfohlen": r["id"] == best, "aus_mp3": klang.falsche_verlustfrei(r)}
                      for r in ms]
        mitglieder.sort(key=lambda m: not m["empfohlen"])
        erg.append({"id": g[0], "sicher": bool(g[1]), "mitglieder": mitglieder})
    return erg


class DupEntscheidung(BaseModel):
    status: str = Field(pattern="^(freigegeben|verworfen)$")
    mitglieder: list[int] = Field(max_length=200)
    behalten: Optional[int] = None


@app.post("/api/duplikate/{gid}")
def api_dup_entscheiden(gid: int, e: DupEntscheidung, request: Request):
    c = con()
    c.execute("BEGIN IMMEDIATE")
    try:
        behalten = duplikate.entscheiden(c, gid, e.status, e.mitglieder, e.behalten, von="ich", wer=_wer(request))
    except duplikate.Abgelehnt as a:
        c.rollback()
        raise HTTPException(a.code, a.text)
    c.commit()
    return {"ok": True, "behalten": behalten}


@app.post("/api/duplikate/{gid}/zuruecknehmen")
def api_dup_zuruecknehmen(gid: int, request: Request):
    c = con()
    c.execute("BEGIN IMMEDIATE")
    try:
        neu = duplikate.zuruecknehmen(c, gid, wer=_wer(request))
    except duplikate.Abgelehnt as a:
        c.rollback()
        raise HTTPException(a.code, a.text)
    c.commit()
    return {"status": neu, "neu_vorgelegt": neu == "ersetzt"}


@app.get("/api/duplikate-auto")
def api_dup_auto():
    return {"stufen": dupauto.uebersicht(con())}


class AutoZurueck(BaseModel):
    stufe: Optional[str] = Field(None, pattern="^(1|2|A|B|C|D)$")


@app.post("/api/duplikate-auto/zuruecknehmen")
def api_dup_auto_zurueck(z: AutoZurueck, request: Request):
    c = con()
    c.execute("BEGIN IMMEDIATE")
    n = dupauto.alle_zuruecknehmen(c, z.stufe, wer=_wer(request))
    c.commit()
    return {"zurueckgenommen": n}


# ---- Seiten und Dateien

def _datei(name, typ=None):
    pfad = STATIC / name
    if not pfad.is_file():
        raise HTTPException(404)
    return FileResponse(pfad, media_type=typ, headers={"Cache-Control": "no-cache"})


@app.get("/")
def startseite():
    return _datei("index.html", "text/html; charset=utf-8")


@app.get("/status")
def statusseite():
    return _datei("status.html", "text/html; charset=utf-8")


@app.get("/tokens.css")
def tokens():
    return _datei("tokens.css", "text/css")


@app.get("/favicon.png")
def favicon():
    return _datei("favicon.png", "image/png")


@app.get("/apple-touch-icon.png")
def touch_icon():
    return _datei("apple-touch-icon.png", "image/png")


@app.get("/fa/{teil}/{name}")
def fontawesome(teil: str, name: str):
    if teil not in ("css", "webfonts") or "/" in name or name.startswith("."):
        raise HTTPException(404)
    pfad = STATIC / "fa" / teil / name
    if not pfad.is_file():
        raise HTTPException(404)
    return FileResponse(pfad, headers={"Cache-Control": "public, max-age=604800"})
