"""Genre-Baum pflegen: Namensabgleich, Umbenennen, Verschieben, Zusammenführen.

Eigene Namen sind erlaubt. Der Abgleich soll nur verhindern, dass dasselbe
zweimal unter leicht anderem Namen entsteht („Hardtechno“ / „Hard Techno“).
"""
import difflib
import json
import re
import unicodedata

from . import config

_discogs = None


def norm(name):
    """Vergleichsform: klein, ohne Akzente, Leer- und Satzzeichen; ä→ae usw."""
    s = name.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s)


def aehnlich(a, b):
    na, nb = norm(a), norm(b)
    return bool(na) and (na == nb or difflib.SequenceMatcher(None, na, nb).ratio() >= 0.8)  # „Tecnho“ ~ „Techno“


def discogs_stile():
    """[(stil, obergenre), …] aus dem Discogs-Modell, z. B. ("Dub Techno", "Electronic")."""
    global _discogs
    if _discogs is None:
        try:
            klassen = json.loads(config.DISCOGS400_JSON.read_text())["classes"]
        except OSError:
            klassen = []
        _discogs = [(k.split("---")[1], k.split("---")[0]) for k in klassen if "---" in k]
    return _discogs


def _treffer(q, name):
    """Passt name zur Eingabe q? Teilwort oder ähnlich geschrieben."""
    nq, nn = norm(q), norm(name)
    if not nq:
        return 0.0
    if nq in nn:
        return 1.0 + len(nq) / max(len(nn), 1)  # Teilwort: kürzere Namen zuerst
    return difflib.SequenceMatcher(None, nq, nn).ratio()


def namen_vorschlagen(con, q, pfade, grenze=8):
    """Bestehende Genres und Discogs-Stile zur Eingabe, plus Warnung bei Fast-Dubletten."""
    vorhanden = sorted(((r["id"], r["name"], _treffer(q, r["name"]))
                        for r in con.execute("SELECT id, name FROM genres")), key=lambda x: -x[2])
    vorhanden = [(i, n) for i, n, t in vorhanden if t >= 0.7][:grenze]
    bekannt = {norm(n) for _, n in vorhanden}
    stile = sorted(((s, o, _treffer(q, s)) for s, o in discogs_stile()), key=lambda x: -x[2])
    stile = [(s, o) for s, o, t in stile if t >= 0.7 and norm(s) not in bekannt][:grenze]
    warnung = next((f"„{n}“ gibt es schon" for i, n in vorhanden if aehnlich(q, n)), None)
    return {
        "vorhanden": [{"id": i, "name": n, "pfad": pfade.get(i, n)} for i, n in vorhanden],
        "discogs": [{"name": s, "obergenre": o} for s, o in stile],
        "warnung": warnung,
    }


def _nachfahren(con, gid):
    erg, offen = set(), [gid]
    while offen:
        for (k,) in con.execute("SELECT id FROM genres WHERE eltern_id=?", (offen.pop(),)):
            if k not in erg:
                erg.add(k)
                offen.append(k)
    return erg


class GenreFehler(ValueError):
    pass


def aendern(con, gid, name, eltern_id, _commit=True):
    """Umbenennen und/oder verschieben. Bewertungen bleiben am Genre hängen."""
    name = " ".join((name or "").split())
    if not name or len(name) > 60:
        raise GenreFehler("Name fehlt oder ist zu lang")
    if "/" in name:
        raise GenreFehler("Bitte ohne „/“ (wird später ein Ordnername)")
    if eltern_id == gid or eltern_id in _nachfahren(con, gid):
        raise GenreFehler("Ein Genre kann nicht unter sich selbst hängen")
    # gleiche Schreibweise bis auf Groß/Klein, Leer- und Bindestriche gilt als derselbe Name
    for r in con.execute("SELECT id, name FROM genres WHERE eltern_id IS ? AND id!=?", (eltern_id, gid)):
        if norm(r[1]) == norm(name):
            raise GenreFehler(f"„{r[1]}“ gibt es dort schon. Zusammenführen?")
    con.execute("UPDATE genres SET name=?, eltern_id=? WHERE id=?", (name, eltern_id, gid))
    if _commit:
        con.commit()


def zusammenfuehren(con, quelle, ziel, _commit=True):
    """quelle geht in ziel auf: Bewertungen und Unter-Genres wandern mit, quelle verschwindet."""
    if quelle == ziel:
        raise GenreFehler("Gleiches Genre")
    if ziel in _nachfahren(con, quelle):
        raise GenreFehler("Ziel liegt unterhalb der Quelle; erst verschieben")
    for e in con.execute("SELECT lied_id, haupt, quelle, ts FROM etiketten WHERE genre_id=?", (quelle,)).fetchall():
        vorhanden = con.execute("SELECT haupt FROM etiketten WHERE lied_id=? AND genre_id=?",
                                (e["lied_id"], ziel)).fetchone()
        if vorhanden:  # Lied hatte beide: eine Zeile, Hauptgenre gewinnt
            con.execute("UPDATE etiketten SET haupt=max(haupt, ?) WHERE lied_id=? AND genre_id=?",
                        (e["haupt"], e["lied_id"], ziel))
            con.execute("DELETE FROM etiketten WHERE lied_id=? AND genre_id=?", (e["lied_id"], quelle))
        else:
            con.execute("UPDATE etiketten SET genre_id=? WHERE lied_id=? AND genre_id=?",
                        (ziel, e["lied_id"], quelle))
    # Quelle aus dem Weg räumen: ein gleichnamiges Kind darf gleich an ihre Stelle
    con.execute("UPDATE genres SET name=? WHERE id=?", (f"\x00weg{quelle}", quelle))
    for kind in con.execute("SELECT id, name FROM genres WHERE eltern_id=?", (quelle,)).fetchall():
        # gleichnamiges Kind unter dem Ziel, gleiche Normalform zählt (nie die Quelle selbst:
        # Zusammenführen ins Elternteil) – sonst entstünden Geschwister „Hard Techno“/„hardtechno“
        gleich = next((r for r in con.execute("SELECT id, name FROM genres WHERE eltern_id=? AND id!=?", (ziel, quelle))
                       if norm(r[1]) == norm(kind["name"])), None)
        if gleich:
            zusammenfuehren(con, kind["id"], gleich["id"], _commit=False)
        else:
            con.execute("UPDATE genres SET eltern_id=? WHERE id=?", (ziel, kind["id"]))
    for tabelle, spalte in (("vorhersagen", "genre_id"), ("vorhersagen", "zweit_id"), ("startbeispiele", "genre_id"),
                            ("stichproben", "vorhergesagt"), ("stichproben", "gewaehlt")):
        con.execute(f"UPDATE {tabelle} SET {spalte}=? WHERE {spalte}=?", (ziel, quelle))
    # Schwellen müssen neu geschätzt werden; ein gerade laufendes Training verwirft sein Ergebnis
    con.execute("DELETE FROM modell_info WHERE schluessel='cv_bewertungen'")
    con.execute("INSERT INTO modell_info VALUES ('genre_stand', '1') ON CONFLICT(schluessel) "
                "DO UPDATE SET wert = CAST(wert AS INTEGER) + 1")
    con.execute("DELETE FROM modell_stand WHERE genre_id=?", (quelle,))
    con.execute("DELETE FROM verwechslungen WHERE a=? OR b=?", (quelle, quelle))
    con.execute("DELETE FROM genres WHERE id=?", (quelle,))
    if _commit:
        con.commit()
