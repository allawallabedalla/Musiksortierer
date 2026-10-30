"""Favoriten-Stern in die Musikdatei schreiben, damit er auch außerhalb der App auffindbar ist.

Je Format zwei Merkmale:
- Bewertung 5 Sterne, wo es dafür einen verbreiteten Standard gibt:
  MP3/WAV/AIFF: ID3 POPM (wie Windows Media Player; Windows-Explorer, MusicBee, MediaMonkey, foobar lesen das),
  FLAC/Ogg/Opus: RATING=5, WMA: WM/SharedUserRating=99.
- Text „Favorit“ in der Gruppierung (Grouping): ID3 TIT1, Vorbis GROUPING, MP4 ©grp, WMA WM/ContentGroupDescription –
  durchsuchbar in fast jedem Player. Bestehender Text bleibt, „Favorit“ wird angehängt bzw. entfernt.

Außerdem die erkannte Gesangssprache (sprache.py, ab SPRACHE_DATEI_P Sicherheit) als ISO-639-2-Kürzel
(„deu“, „eng“ …): ID3 TLAN, Vorbis LANGUAGE, MP4 ----:com.apple.iTunes:LANGUAGE, WMA WM/Language.

Und das Tempo (TempoCNN, scan.py) als ganze BPM: ID3 TBPM, Vorbis BPM, MP4 tmpo, WMA WM/BeatsPerMinute.
Vorhandene BPM-Tags werden überschrieben (Wunsch des Nutzers 30.10.2026); geschrieben wird nur, was
das aktuelle Tempo-Modell gemessen hat (tempo_stand), nie Werte des alten Schätzers.

Danach werden Größe/Änderungszeit in der DB nachgeführt: die nächtliche Suche hält die Datei sonst für
geändert (neu analysieren, Duplikat-Entscheidungen ungültig).

  python -m musiksort.tags            # alle ausstehenden Sterne, Sprachen und BPM schreiben (Freigabe muss beschreibbar sein)
  python -m musiksort.tags --zaehlen  # nur zählen
"""
import errno
import os
import sys

from . import config, db, sprache as sprachlib

WORT = "Favorit"
POPM_EMAIL = "Windows Media Player 9 Series"


class NurLesend(RuntimeError):
    pass


def _gruppierung(alt, an):
    teile = [t.strip() for t in (alt or "").replace(";", "/").split("/") if t.strip()]
    teile = [t for t in teile if t.lower() != WORT.lower()]
    if an:
        teile.append(WORT)
    return " / ".join(teile)


def schreiben(pfad, an=None, sprache=None, bpm=None):
    """Stern setzen/entfernen (an=None: unverändert), Sprache (ISO 639-2) und BPM (ganze Zahl) setzen (None: unverändert).

    Rückgabe: Format-Kürzel. Wirft NurLesend bei schreibgeschützter Freigabe."""
    import mutagen
    from mutagen.asf import ASF
    from mutagen.mp4 import MP4
    try:
        f = mutagen.File(pfad)
        if f is None:
            raise ValueError("Format unbekannt")
        if isinstance(f, MP4):
            from mutagen.mp4 import MP4FreeForm
            if f.tags is None:
                f.add_tags()
            if an is not None:
                alt = (f.tags.get("\xa9grp") or [""])[0]
                f.tags["\xa9grp"] = [_gruppierung(alt, an)] if _gruppierung(alt, an) else []
                if not f.tags["\xa9grp"]:
                    del f.tags["\xa9grp"]
            if sprache:
                f.tags["----:com.apple.iTunes:LANGUAGE"] = [MP4FreeForm(sprache.encode())]
            if bpm:
                f.tags["tmpo"] = [bpm]
            art = "mp4"
        elif isinstance(f, ASF):
            if an is not None:
                alt = str((f.tags.get("WM/ContentGroupDescription") or [""])[0])
                neu = _gruppierung(alt, an)
                if neu:
                    f.tags["WM/ContentGroupDescription"] = [neu]
                elif "WM/ContentGroupDescription" in f.tags:
                    del f.tags["WM/ContentGroupDescription"]
                if an:
                    f.tags["WM/SharedUserRating"] = [99]
                elif "WM/SharedUserRating" in f.tags:
                    del f.tags["WM/SharedUserRating"]
            if sprache:
                f.tags["WM/Language"] = [sprache]
            if bpm:
                f.tags["WM/BeatsPerMinute"] = [str(bpm)]
            art = "wma"
        elif hasattr(f, "tags") and f.tags is not None and hasattr(f.tags, "getall"):  # ID3 (MP3, WAV, AIFF)
            art = _id3(f.tags, an, sprache, bpm)
        elif hasattr(f, "tags") and f.tags is None and _ist_id3_format(f):
            f.add_tags()
            art = _id3(f.tags, an, sprache, bpm)
        else:  # Vorbis-Kommentare (FLAC, Ogg, Opus)
            if f.tags is None:
                f.add_tags()
            if an is not None:
                alt = (f.tags.get("GROUPING") or [""])[0]
                neu = _gruppierung(alt, an)
                if neu:
                    f.tags["GROUPING"] = neu
                elif "GROUPING" in f.tags:
                    del f.tags["GROUPING"]
                if an:
                    f.tags["RATING"] = "5"
                elif "RATING" in f.tags:
                    del f.tags["RATING"]
            if sprache:
                f.tags["LANGUAGE"] = sprache
            if bpm:
                f.tags["BPM"] = str(bpm)
            art = "vorbis"
        f.save()
        return art
    except OSError as e:
        if e.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise NurLesend(str(e)) from e
        raise
    except mutagen.MutagenError as e:  # mutagen verpackt OSError
        ursache = e.__cause__ or (e.args[0] if e.args else None)
        if isinstance(ursache, OSError) and ursache.errno in (errno.EROFS, errno.EACCES, errno.EPERM):
            raise NurLesend(str(ursache)) from e
        raise


def _ist_id3_format(f):
    from mutagen.aiff import AIFF
    from mutagen.mp3 import MP3
    from mutagen.wave import WAVE
    return isinstance(f, (MP3, WAVE, AIFF))


def _id3(tags, an, sprache=None, bpm=None):
    from mutagen.id3 import POPM, TBPM, TIT1, TLAN
    if an is not None:
        alt = str(tags["TIT1"]) if "TIT1" in tags else ""
        neu = _gruppierung(alt, an)
        tags.delall("TIT1")
        if neu:
            tags.add(TIT1(encoding=3, text=[neu]))
        tags.delall(f"POPM:{POPM_EMAIL}")
        if an:
            tags.add(POPM(email=POPM_EMAIL, rating=255, count=0))
    if sprache:
        tags.delall("TLAN")
        tags.add(TLAN(encoding=3, text=[sprache]))
    if bpm:
        tags.delall("TBPM")
        tags.add(TBPM(encoding=3, text=[str(bpm)]))
    return "id3"


def lesen(pfad):
    """True, wenn die Datei den Favoriten-Text in der Gruppierung trägt (für Prüfungen)."""
    import mutagen
    f = mutagen.File(pfad)
    if f is None or f.tags is None:
        return False
    for k in ("TIT1", "GROUPING", "\xa9grp", "WM/ContentGroupDescription"):
        try:
            werte = f.tags[k]
        except (KeyError, ValueError, TypeError):
            continue
        text = " ".join(str(w) for w in (werte if isinstance(werte, list) else [werte]))
        if WORT.lower() in text.lower():
            return True
    return False


def sprache_iso(code, p):
    """ISO-Kürzel für die Datei, nur bei sicherer Erkennung einer bekannten Sprache."""
    return sprachlib.ISO.get(code) if p is not None and p >= config.SPRACHE_DATEI_P else None


def ausstehend(con):
    """[(id, pfad, stern oder None, sprache oder None, bpm oder None)]: was sich gegenüber der Datei geändert hat."""
    erg = []
    for r in con.execute("SELECT id, pfad, stern, stern_in_datei, sprache, sprache_p, sprache_in_datei, "
                         "bpm, tempo_stand, bpm_in_datei FROM lieder "
                         "WHERE status='analysiert' AND (stern != coalesce(stern_in_datei, 0) "
                         "OR sprache_p >= ? OR (tempo_stand=? AND bpm > 0))",
                         (config.SPRACHE_DATEI_P, config.TEMPO_STAND)):
        stern = r["stern"] if r["stern"] != (r["stern_in_datei"] or 0) else None
        iso = sprache_iso(r["sprache"], r["sprache_p"])
        iso = iso if iso and iso != r["sprache_in_datei"] else None
        bpm = round(r["bpm"]) if r["tempo_stand"] == config.TEMPO_STAND and r["bpm"] else None
        bpm = bpm if bpm and bpm != r["bpm_in_datei"] else None
        if stern is not None or iso or bpm:
            erg.append((r["id"], r["pfad"], stern, iso, bpm))
    return erg


def alle_schreiben(con=None, ausgabe=print):
    """Alle ausstehenden Sterne, Sprachen und BPM in die Dateien. Bricht bei schreibgeschützter Freigabe sofort ab."""
    con = con or db.verbinden()
    n = fehler = 0
    for lid, rel, stern, iso, bpm in ausstehend(con):
        pfad = config.medienpfad(rel)
        try:
            schreiben(pfad, None if stern is None else bool(stern), iso, bpm)
        except NurLesend:
            ausgabe(f"Freigabe ist nur lesend eingebunden: nichts geschrieben ({n} vorher geschrieben)")
            return n
        except Exception as e:  # einzelne kaputte Datei: weiter mit den anderen
            fehler += 1
            ausgabe(f"{rel}: {e!r}")
            continue
        st = os.stat(pfad)
        with con:  # Größe/Zeit nachführen, sonst gilt die Datei nachts als geändert
            if stern is not None:
                con.execute("UPDATE lieder SET stern_in_datei=? WHERE id=?", (stern, lid))
            if iso:
                con.execute("UPDATE lieder SET sprache_in_datei=? WHERE id=?", (iso, lid))
            if bpm:
                con.execute("UPDATE lieder SET bpm_in_datei=? WHERE id=?", (bpm, lid))
            con.execute("UPDATE lieder SET groesse=?, mtime=?, sha256=NULL, sha_stand=NULL WHERE id=?",
                        (st.st_size, st.st_mtime, lid))
        n += 1
    ausgabe(f"{n} Dateien geschrieben, {fehler} Fehler")
    return n


if __name__ == "__main__":
    if sys.argv[1:] == ["--zaehlen"]:
        a = ausstehend(db.verbinden())
        print(sum(x[2] is not None for x in a), "Sterne,", sum(bool(x[3]) for x in a), "Sprachen und",
              sum(bool(x[4]) for x in a), "BPM warten darauf, in die Datei geschrieben zu werden")
    else:
        alle_schreiben()
