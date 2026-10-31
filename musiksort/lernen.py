"""Lernmodell: aus deinen Bewertungen Genres vorhersagen, Sicherheit schätzen,
das Backlog sinnvoll ordnen.

Der Klang-Steckbrief (Discogs-EffNet) bleibt fest; gelernt wird nur eine
logistische Regression auf diesen Steckbriefen.

Automatisch einsortiert wird ein Lied nur, wenn alles drei passt:
1. Sicherheit über der Schwelle seines Genres (Kreuzvalidierung, vorsichtige
   Wilson-Untergrenze ≥ Ziel),
2. es ähnelt den bewerteten Liedern dieses Genres wirklich (sonst ist es
   vermutlich ein Genre, das es noch gar nicht gibt),
3. die Stichproben widersprechen dem Genre nicht.

Physisch verschoben (späterer Sortierlauf) wird nur, was zusätzlich POSITIV belegt ist:
≥ 10 Stichproben des vorhergesagten Genres, vorsichtige Untergrenze ≥ 90 % (Spalte
modell_stand.belegt). Grund: Fehlt im Baum ein Geschwister-Genre, schluckt das bekannte
dessen Lieder, ohne dass die Kreuzvalidierung es merken kann (Prüfrunde 2C, Simulation).

  python -m musiksort.lernen cluster      # Klanggruppen neu berechnen (eigener Prozess)
  python -m musiksort.lernen trainieren
"""
import collections
import json
import os
import random
import sqlite3
import subprocess
import sys
import threading
import time

import numpy as np

from . import config, db, duplikate, sprache

_sperre = threading.RLock()
# Ändert sich, was das Modell sieht, sind alte Schwellen ungültig (Kreuzvalidierung neu)
MERKMALE = "klang+laenge+sprache-1/familien/C8"  # „/familien“: Kreuzvalidierung hält Kopien zusammen; C8: config.MODELL_C


class Bestand:
    """Steckbriefe aller analysierten Lieder im Speicher (float16, normiert, ~320 MB bei 125k)."""

    def __init__(self):
        self.ids = np.zeros(0, dtype=np.int64)
        self.X = np.zeros((0, 1280), dtype=np.float16)
        self.dauer = np.zeros(0, dtype=np.float32)  # Sekunden, für die Längen-Merkmale
        self.sprache = np.zeros((0, sprache.BREITE), dtype=np.float32)  # Gesangssprache (sprache.py)
        self.Z = np.zeros((0, 1280), dtype=np.float16)  # Zeitverlauf, standardisiert (0 = nicht gemessen)
        self.hat_z = np.zeros(0, dtype=bool)
        self._zeitverlauf = -1
        self.pos = {}
        self._sprachen = -1
        self._stand = -1
        self._lade_sperre = threading.Lock()

    def aktualisieren(self, con, erzwingen=False, brauche=()):
        """Neu laden bei spürbarer Änderung oder wenn gebrauchte Lieder fehlen.

        Geladen wird ohne die Anfrage-Sperre; nur der Austausch ist kurz gesperrt.
        """
        with self._lade_sperre:
            n, n_spr, n_z = con.execute("SELECT COUNT(*), COUNT(sprache_probs), COUNT(emb_std) FROM lieder "
                                        "WHERE status='analysiert' AND emb IS NOT NULL").fetchone()
            fehlt = any(i not in self.pos for i in brauche)
            # neu gemessene Sprachen zählen wie neue Lieder (ändern die Merkmale)
            if not erzwingen and not fehlt and self._stand >= 0 and abs(n - self._stand) <= max(20, 0.02 * self._stand) \
                    and abs(n_spr - self._sprachen) <= max(20, 0.02 * max(self._stand, 1)) \
                    and abs(n_z - self._zeitverlauf) <= max(20, 0.02 * max(self._stand, 1)):
                return
            ids = np.zeros(n, dtype=np.int64)
            X = np.zeros((n, 1280), dtype=np.float16)
            dauer = np.zeros(n, dtype=np.float32)
            spr = np.zeros((n, sprache.BREITE), dtype=np.float32)
            Z = np.zeros((n, 1280), dtype=np.float16)
            hat_z = np.zeros(n, dtype=bool)
            k = 0
            cur = con.execute("SELECT id, emb, dauer, sprache_probs, emb_std FROM lieder "
                              "WHERE status='analysiert' AND emb IS NOT NULL")
            while k < n:
                block = cur.fetchmany(5000)
                if not block:
                    break
                B = np.stack([np.frombuffer(r[1], np.float16) for r in block]).astype(np.float32)
                B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
                m = min(len(block), n - k)
                ids[k:k + m] = [r[0] for r in block[:m]]
                X[k:k + m] = B[:m]
                dauer[k:k + m] = [r[2] or 0 for r in block[:m]]
                spr[k:k + m] = sprache.merkmale([r[3] for r in block[:m]])
                for j, r in enumerate(block[:m]):
                    if r[4] is not None:
                        Z[k + j] = np.frombuffer(r[4], np.float16)
                        hat_z[k + j] = True
                k += m
            ids, X, dauer, spr, Z, hat_z = ids[:k], X[:k], dauer[:k], spr[:k], Z[:k], hat_z[:k]
            _zeitverlauf_standardisieren(Z, hat_z)
            with _sperre:
                self.ids, self.X, self.dauer, self.sprache, self._stand = ids, X, dauer, spr, k
                self.Z, self.hat_z, self._zeitverlauf = Z, hat_z, int(hat_z.sum())
                self._sprachen = int(spr[:, -1].sum() / max(config.SPRACHE_GEWICHT, 1e-9))
                self.pos = {int(i): j for j, i in enumerate(ids)}

    def mal(self, M, X=None):
        """X @ M.T in Blöcken (float16 × float32 ohne Riesenkopie)."""
        X = self.X if X is None else X
        M = np.asarray(M, dtype=np.float32)
        erg = np.empty((len(X), M.shape[0]), dtype=np.float32)
        for s in range(0, len(X), 10000):
            erg[s:s + 10000] = X[s:s + 10000].astype(np.float32) @ M.T
        return erg

    def aehnliche(self, lied_id, n=20):
        with _sperre:
            ids, X, pos = self.ids, self.X, self.pos
        k = pos.get(lied_id)
        if k is None:
            return []
        s = self.mal(X[k:k + 1], X)[:, 0]
        s[k] = -1
        best = np.argpartition(-s, min(n, len(s) - 1))[:n]
        best = best[np.argsort(-s[best])]
        return [(int(ids[i]), float(s[i])) for i in best]


bestand = Bestand()


def wilson_unten(treffer, n, z=None):
    """Vorsichtige Untergrenze einer Trefferquote (z=1,28 ≈ 90 % einseitig)."""
    z = config.WILSON_Z if z is None else z
    if n == 0:
        return 0.0
    p = treffer / n
    mitte = p + z * z / (2 * n)
    rand = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((mitte - rand) / (1 + z * z / n))


def wilson_oben(treffer, n, z=None):
    """Obergrenze: liegt sie unter dem Ziel, ist das Genre nachweislich zu ungenau."""
    return 1.0 - wilson_unten(n - treffer, n, z)


def schwelle_finden(sicherheit, richtig, ziel=None, min_n=5):
    """Niedrigste Sicherheit, ab der die vorsichtige Trefferquote das Ziel hält.

    Eingabe: alle Kreuzvalidierungs-Vorhersagen eines Genres. Mit z=1,28 braucht
    „≥ 95 %“ mindestens 32 richtige Fälle ohne Fehler; darunter gibt es keine Schwelle.
    Rückgabe (schwelle, beobachtete Quote, n) oder None.
    """
    ziel = config.ZIEL_TREFFERQUOTE if ziel is None else ziel
    ordnung = np.argsort(-np.asarray(sicherheit))
    s, r = np.asarray(sicherheit)[ordnung], np.asarray(richtig)[ordnung]
    treffer = np.cumsum(r)
    bestes = None
    for i in range(min_n - 1, len(s)):
        if wilson_unten(int(treffer[i]), i + 1) >= ziel:
            bestes = (float(s[i]), float(treffer[i] / (i + 1)), i + 1)
    return bestes


def laenge_merkmale(dauer):
    """Länge als Merkmal: Mixtapes klingen wie ihr Genre, erkennbar sind sie an der Dauer.

    Vier Spalten: log-Dauer (0 bei 4 min, +1 je Vervierfachung, begrenzt) und
    über 10 / 20 / 45 Minuten. Fehlende Dauer zählt wie ein 4-Minuten-Lied.
    """
    d = np.asarray(dauer, dtype=np.float64)
    d = np.where(np.isfinite(d) & (d > 0), d, 240.0)
    d = np.clip(d, 10, 6 * 3600)
    return (np.column_stack([np.clip(np.log2(d / 240) / 2, -2, 3), d > 600, d > 1200, d > 2700])
            .astype(np.float32) * config.LAENGE_GEWICHT)


def _zeitverlauf_standardisieren(Z, hat):
    """In place: je Spalte z-standardisiert über die gemessenen Lieder, Blocklänge ~1 wie der normierte
    Steckbrief. Nicht gemessene Lieder bleiben 0 = Durchschnitt (neutral)."""
    if not hat.any():
        return
    zeilen = np.flatnonzero(hat)
    summe, quadrat = np.zeros(Z.shape[1]), np.zeros(Z.shape[1])
    for s in range(0, len(zeilen), 20000):
        t = Z[zeilen[s:s + 20000]].astype(np.float64)
        summe += t.sum(0)
        quadrat += (t ** 2).sum(0)
    mittel = summe / len(zeilen)
    sd = np.sqrt(np.maximum(quadrat / len(zeilen) - mittel ** 2, 0)) + 1e-6
    for s in range(0, len(zeilen), 20000):
        z = zeilen[s:s + 20000]
        Z[z] = ((Z[z].astype(np.float32) - mittel) / sd / np.sqrt(Z.shape[1])).astype(np.float16)


def _merkmale(emb, dauer, spr=None, zeit=None):
    """Was das Genre-Modell sieht: Klang-Steckbrief + Länge + Gesangssprache (Nullen = nicht gemessen),
    mit zeit (standardisierter Zeitverlauf) zusätzlich dieser Block."""
    if spr is None:
        spr = np.zeros((len(emb), sprache.BREITE), dtype=np.float32)
    teile = [emb.astype(np.float32), laenge_merkmale(dauer), spr]
    if zeit is not None:
        teile.append(zeit.astype(np.float32) * config.ZEITVERLAUF_GEWICHT)
    return np.hstack(teile)


def _etiketten(con):
    return con.execute("SELECT e.lied_id, e.genre_id, e.quelle FROM etiketten e JOIN genres g ON g.id=e.genre_id "
                       "WHERE e.haupt=1").fetchall()


def _baum(con):
    """{genre: Menge aus ihm selbst und allen Unter-Genres} (Kreis-sicher)."""
    eltern = dict(con.execute("SELECT id, eltern_id FROM genres").fetchall())
    teil = {g: {g} for g in eltern}
    for g in eltern:
        e, gesehen = eltern[g], {g}
        while e is not None and e in eltern and e not in gesehen:
            teil[e].add(g)
            gesehen.add(e)
            e = eltern[e]
    return teil, eltern


def _stichproben(con, teil=None):
    """Je GEPRÜFTEM Genre: (richtig, gesamt). Richtig, wenn deine Wahl das Genre oder eines seiner
    Unter-Genres ist (ein Lied im Ordner „Hip Hop“, das du „Gangsta“ nennst, lag nicht falsch).
    So wird das Genre belastet, das falsch sortiert hätte."""
    teil = teil or _baum(con)[0]
    erg = {}
    for vorh, gew in con.execute("SELECT vorhergesagt, gewaehlt FROM stichproben"):
        ok, n = erg.get(vorh, (0, 0))
        erg[vorh] = (ok + int(gew in teil.get(vorh, {vorh})), n + 1)
    return erg


def stichproben_fehlen(ok, n, grenze=500):
    """Wie viele richtige Stichproben noch fehlen, bis das Genre belegt ist (0 = belegt, None = mehr als grenze)."""
    m = max(n, config.STICHPROBE_MIN)
    while wilson_unten(ok + (m - n), m) < config.STICHPROBE_ZIEL:
        m += 1
        if m - n > grenze:
            return None
    return m - n


def stichproben_gewichte(wartend, stich):
    """Welches Genre die nächste Stichprobe bekommt: Lieder, die eine richtige Antwort freischaltet.

    wartend: Genre → Zahl automatisch zugeordneter Lieder; stich: Genre → (richtig, gesamt).
    Gewicht = wartende Lieder / fehlende Stichproben – ein großes Genre kurz vor dem Beleg zuerst.
    Belegte Genres werden nur noch selten nachgeprüft, aussichtslose ebenso (wie fehlten STICHPROBE_NACHPRUEFEN)."""
    paare = []
    for g, n in wartend.items():
        fehlt = stichproben_fehlen(*stich.get(g, (0, 0)))
        paare.append((g, n / (fehlt or config.STICHPROBE_NACHPRUEFEN)))
    return paare


def _info(con, schluessel, wert=None):
    if wert is None:
        r = con.execute("SELECT wert FROM modell_info WHERE schluessel=?", (schluessel,)).fetchone()
        return r[0] if r else None
    con.execute("INSERT OR REPLACE INTO modell_info VALUES (?,?)", (schluessel, str(wert)))


def _kreuzvalidierung(X, y, zaehl, familien=None):
    """Ehrliche Schätzung je Genre: 5-fach über alle Genres mit ≥ 5 Beispielen.

    familien (je Zeile): Kopien desselben Liedes landen immer im selben Teil. Sonst lernt das Modell
    an der einen Kopie und „erkennt“ die andere bei der Prüfung wieder – die Quote sähe zu gut aus.

    Lieder kleinerer Genres werden vom Modell der großen vorhergesagt und zählen dort
    als Fehler: so fließt „das Genre gibt es (noch) kaum“ in die Schwellen mit ein.
    Rückgabe: (wahr, vorhergesagt, sicherheit, P, klassen, reihenfolge) über alle bewerteten Lieder;
    P = alle Wahrscheinlichkeiten (für die Obergenre-Stufe), reihenfolge = Zeilen in X.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    gross = np.array([zaehl[g] >= 5 for g in y])
    if len(set(y[gross])) < 2:
        return None
    modell = LogisticRegression(C=config.MODELL_C, max_iter=2000, class_weight="balanced")
    if familien is None:
        teile, gruppen = StratifiedKFold(5, shuffle=True, random_state=0), None
    else:
        teile, gruppen = StratifiedGroupKFold(5, shuffle=True, random_state=0), np.asarray(familien)[gross]
    p = cross_val_predict(modell, X[gross], y[gross], groups=gruppen, cv=teile, method="predict_proba")
    klassen = np.unique(y[gross])
    wahr, vorh, sicher, alle_p = [y[gross]], [klassen[p.argmax(1)]], [p.max(1)], [p]
    reihenfolge = [np.flatnonzero(gross)]
    if (~gross).any():
        modell.fit(X[gross], y[gross])
        p2 = modell.predict_proba(X[~gross])
        wahr.append(y[~gross])
        vorh.append(modell.classes_[p2.argmax(1)])
        sicher.append(p2.max(1))
        alle_p.append(p2)
        reihenfolge.append(np.flatnonzero(~gross))
    return (np.concatenate(wahr), np.concatenate(vorh), np.concatenate(sicher), np.vstack(alle_p), klassen,
            np.concatenate(reihenfolge))


def _naehe_grenze(Eg):
    """Ausreißer-Grenze: Nähe zum Zentrum, ohne das Lied selbst (Leave-one-out), 10-%-Quantil."""
    summe = Eg.sum(0)
    ohne = summe[None, :] - Eg
    naehe = (ohne * Eg).sum(1) / (np.linalg.norm(ohne, axis=1) + 1e-9)
    return float(np.quantile(naehe, config.AUSREISSER_QUANTIL))


def ober_schwellen(wahr, vorh, P, klassen, teil, zaehl):
    """Obergenre-Stufe: je Genre mit Unter-Genres die Schwelle für „sicher irgendwo in diesem Teilbaum“.

    Sicherheit = Summe der Wahrscheinlichkeiten aller Genres im Teilbaum; betrachtet werden die Lieder,
    deren wahrscheinlichstes Genre im Teilbaum liegt; richtig = dein Genre liegt im Teilbaum.
    Gleiche Hürde wie bei einzelnen Genres (vorsichtig ≥ Ziel, ≥ 32 Bewertungen im Teilbaum).
    Rückgabe {genre: (schwelle, quote, n)}.
    """
    erg = {}
    index = {int(k): j for j, k in enumerate(klassen)}
    for a, d in teil.items():
        spalten = [index[g] for g in d if g in index]
        if len(d) < 2 or len(spalten) < 2 or sum(zaehl.get(g, 0) for g in d) < config.MIN_BEWERTUNGEN_AUTO:
            continue
        maske = np.isin(vorh, list(d))
        if maske.sum() < 5:
            continue
        s = schwelle_finden(P[maske][:, spalten].sum(1), np.isin(wahr[maske], list(d)))
        if s:
            erg[a] = (max(s[0], 0.5), s[1], s[2])
    return erg


def _schreiben_beginnen(con, genre_stand):
    """Schreibtransaktion öffnen – oder ablehnen, wenn während des Rechnens Genres
    zusammengeführt wurden (App, anderer Prozess): dann ist das Ergebnis veraltet."""
    if con.in_transaction:  # offene Lesetransaktion des Aufrufers abschließen
        con.commit()
    con.execute("BEGIN IMMEDIATE")
    if _info(con, "genre_stand") != genre_stand:
        con.rollback()
        return False
    return True


def trainieren(con=None, cv_erzwingen=False, trocken=False):
    """Modell neu trainieren und alles vorhersagen.

    Die teure Kreuzvalidierung (Schwellen, Ausreißer-Grenzen) läuft nur alle
    CV_ALLE neuen Bewertungen oder wenn ein Genre die Mindestzahl erreicht; dazwischen
    gelten die letzten Schwellen. Geschrieben wird am Ende in einer kurzen Transaktion.
    trocken=True: nichts schreiben, nur zählen (Rückgabe: dict mit der Wirkung).
    """
    from sklearn.linear_model import LogisticRegression

    con = con or db.verbinden()
    genre_stand = _info(con, "genre_stand")  # ändert sich bei jedem Zusammenführen
    alle_et = _etiketten(con)
    bestand.aktualisieren(con, brauche=[e["lied_id"] for e in alle_et])
    with _sperre:  # werden bei Updates ersetzt, nie verändert
        ids, X_alle, dauer_alle, spr_alle, pos = bestand.ids, bestand.X, bestand.dauer, bestand.sprache, bestand.pos
        Z_alle, hat_z = bestand.Z, bestand.hat_z
    et = [e for e in alle_et if e["lied_id"] in pos]
    zaehl = {}
    for e in et:
        zaehl[e["genre_id"]] = zaehl.get(e["genre_id"], 0) + 1
    # Startmodell (fremde Beispiele): nur fürs Modell, Schwellen und Freigaben bleiben bei den eigenen
    from . import startmodell
    start = startmodell.beispiele(con)
    mit_start = dict(zaehl)
    if start is not None:
        for g in start[3]:
            mit_start[int(g)] = mit_start.get(int(g), 0) + 1
    genres = [g for g, n in mit_start.items() if n >= 2]
    jetzt = db.jetzt()
    teil, eltern = _baum(con)
    stich = _stichproben(con, teil)

    if len(genres) < 2:
        with _sperre:
            if not _schreiben_beginnen(con, genre_stand):
                return False
            for t in ("vorhersagen", "modell_stand", "verwechslungen"):
                con.execute(f"DELETE FROM {t}")
            con.executemany("INSERT INTO modell_stand(genre_id, n, stichprobe_ok, stichprobe_n, ts) VALUES (?,?,?,?,?)",
                            [(g, n, *stich.get(g, (0, 0)), jetzt) for g, n in zaehl.items()])
            con.commit()
        return True
    et = [e for e in et if e["genre_id"] in genres]
    zeilen_et = [pos[e["lied_id"]] for e in et]
    E = X_alle[zeilen_et].astype(np.float32)          # nur Klang: für Zentren und Ausreißer-Grenze
    # Zeitverlauf erst, wenn fast alle bewerteten Lieder ihn haben (sonst lernt das Modell aus Lücken)
    mit_zeit = bool(len(zeilen_et)) and hat_z[zeilen_et].mean() >= config.ZEITVERLAUF_AB
    merkmale = MERKMALE + ("+zeitverlauf" if mit_zeit else "")
    def zeit(sl):
        return Z_alle[sl] if mit_zeit else None
    X = _merkmale(E, dauer_alle[zeilen_et], spr_alle[zeilen_et], zeit(zeilen_et))  # fürs Genre-Modell
    y = np.array([e["genre_id"] for e in et])

    # Kreuzvalidierung nur, wenn sich genug geändert hat
    reif = sorted(g for g, n in zaehl.items() if n >= config.MIN_BEWERTUNGEN_AUTO)
    spr_bewertet = int(spr_alle[zeilen_et, -1].astype(bool).sum())
    letzte_n, letzte_reif = _info(con, "cv_bewertungen"), _info(con, "cv_reif")
    cv_faellig = (cv_erzwingen or letzte_n is None or len(et) - int(letzte_n) >= config.CV_ALLE
                  or letzte_reif != json.dumps(reif) or _info(con, "merkmale") != merkmale
                  # bewertete Lieder mit neu gemessener Sprache: Schwellen neu schätzen
                  or abs(spr_bewertet - int(_info(con, "cv_sprache") or 0)) >= config.CV_ALLE)
    schwellen, quoten, n_sicher, roh, mindest = {}, {}, {}, {}, {}
    ober, ober_mindest = {}, {}   # Obergenre-Stufe: {genre: (schwelle, quote, n)}, Ausreißer-Grenze
    verwechselt = None
    if cv_faellig:
        familie = duplikate.kopien(con)[0]
        eigen = np.array([zaehl[g] >= 2 for g in y], dtype=bool)  # wie bisher: nur eigene Genres mit ≥ 2
        cv = _kreuzvalidierung(X[eigen], y[eigen], zaehl,
                               [familie.get(e["lied_id"], -e["lied_id"]) for e, m in zip(et, eigen) if m]) \
            if eigen.any() else None
        if cv is not None:
            wahr, vorh, sicher, P_cv, klassen_cv, _ = cv
            richtig = wahr == vorh
            for g in np.unique(vorh):
                g = int(g)
                roh[g] = float(richtig[vorh == g].mean())  # wie oft „g“ als Vorhersage stimmte
                if zaehl.get(g, 0) >= config.MIN_BEWERTUNGEN_AUTO:
                    s = schwelle_finden(sicher[vorh == g], richtig[vorh == g])
                    if s:
                        # nie unter 50 %: bei vielen Genres wäre die Schwelle sonst kaum mehr als „am ehesten g“
                        schwellen[g], quoten[g], n_sicher[g] = max(s[0], 0.5), s[1], s[2]
            verwechselt = verwechslungen_finden(wahr, vorh, ab_anteil=config.VERWECHSLUNG_MERKEN)
            # Ausreißer-Grenze: Nähe zum Genre-Zentrum, ohne das Lied selbst (Leave-one-out)
            for g in schwellen:
                mindest[g] = _naehe_grenze(E[y == g])
            ober = ober_schwellen(wahr, vorh, P_cv, klassen_cv, teil, zaehl)
            for a in ober:
                ober_mindest[a] = _naehe_grenze(E[np.isin(y, list(teil[a]))])
    else:  # letzte Schwellen weiterverwenden
        for r in con.execute("SELECT * FROM modell_stand"):
            if r["schwelle"] is not None:
                schwellen[r["genre_id"]], quoten[r["genre_id"]] = r["schwelle"], r["trefferquote"]
                n_sicher[r["genre_id"]], mindest[r["genre_id"]] = r["n_sicher"], r["mindest"]
            if r["quote_roh"] is not None:
                roh[r["genre_id"]] = r["quote_roh"]
            if r["ober_schwelle"] is not None:
                ober[r["genre_id"]] = (r["ober_schwelle"], r["ober_quote"], r["ober_n"])
                ober_mindest[r["genre_id"]] = r["ober_mindest"]
            # widerlegte Genres: Schwellen ruhen, die Stichproben unten entscheiden jedes Mal neu
            if r["pruef_schwelle"] is not None:
                schwellen[r["genre_id"]], mindest[r["genre_id"]] = r["pruef_schwelle"], r["mindest"]
                quoten[r["genre_id"]], n_sicher[r["genre_id"]] = r["trefferquote"], r["n_sicher"]
            if r["pruef_ober_schwelle"] is not None:
                ober[r["genre_id"]] = (r["pruef_ober_schwelle"], r["ober_quote"], r["ober_n"])
                ober_mindest[r["genre_id"]] = r["ober_mindest"]

    # Stichproben: widerlegt → aus; positiv belegt → darf später physisch verschoben werden.
    # Widerlegte Genres bleiben Prüf-Kandidaten (pruef_id): ohne neue Stichproben könnten sie sich nie erholen.
    belegt, pruef_fein, pruef_ober = {}, {}, {}
    for g in set(schwellen) | set(ober):
        ok, n = stich.get(g, (0, 0))
        if n >= config.STICHPROBE_MIN and wilson_oben(ok, n) < config.ZIEL_TREFFERQUOTE:
            if g in schwellen:
                pruef_fein[g] = schwellen.pop(g)
            if g in ober:
                pruef_ober[g] = ober.pop(g)
            continue
        belegt[g] = int(n >= config.STICHPROBE_MIN and wilson_unten(ok, n) >= config.STICHPROBE_ZIEL)

    E_fit, X_fit, y_fit, gewicht = E, X, y, np.ones(len(y))
    if start is not None:
        nimm = np.isin(start[3], genres)
        E_fit = np.vstack([E, start[0][nimm]])
        # Startbeispiele haben keinen Zeitverlauf: 0 = Durchschnitt
        X_fit = np.vstack([X, _merkmale(start[0][nimm], start[1][nimm], start[2][nimm],
                                        np.zeros((int(nimm.sum()), 1280), np.float32) if mit_zeit else None)])
        y_fit = np.concatenate([y, start[3][nimm]])
        gewicht = np.concatenate([gewicht, np.full(int(nimm.sum()), config.START_GEWICHT)])
    modell = LogisticRegression(C=config.MODELL_C, max_iter=2000, class_weight="balanced").fit(X_fit, y_fit, sample_weight=gewicht)
    klassen = [int(k) for k in modell.classes_]
    zentren = np.zeros((len(klassen), E.shape[1]), np.float32)
    for j, g in enumerate(klassen):
        summe = E_fit[y_fit == g].sum(0)
        zentren[j] = summe / (np.linalg.norm(summe) + 1e-9)

    # Obergenre-Stufe: Spalten des Teilbaums und Klang-Zentrum je Obergenre
    index = {g: j for j, g in enumerate(klassen)}
    ober_spalten, ober_zentrum = {}, {}
    for a in set(ober) | set(pruef_ober):
        spalten = [index[g] for g in teil.get(a, {a}) if g in index]
        if len(spalten) < 2:
            continue
        ober_spalten[a] = spalten
        summe = E[np.isin(y, list(teil[a]))].sum(0)
        ober_zentrum[a] = summe / (np.linalg.norm(summe) + 1e-9)

    bewertet = {e["lied_id"] for e in alle_et}
    alle_fein = {**schwellen, **pruef_fein}
    alle_ober = {a: s[0] for a, s in {**ober, **pruef_ober}.items()}
    nur_ober = {a: s[0] for a, s in ober.items()}
    zeilen = []
    for start in range(0, len(ids), 20000):
        teil_x = X_alle[start:start + 20000].astype(np.float32)
        P = modell.predict_proba(_merkmale(teil_x, dauer_alle[start:start + 20000], spr_alle[start:start + 20000],
                                           zeit(slice(start, start + 20000))))
        naehe = teil_x @ zentren.T
        p_ober = {a: P[:, sp].sum(1) for a, sp in ober_spalten.items()}
        n_ober = {a: teil_x @ z for a, z in ober_zentrum.items()}
        top = np.argsort(-P, axis=1)[:, :2]
        def einordnen(k, j1, g1, fein, grob):
            """(auto, ober_id, ober_p) mit diesen Schwellen; selbst bewertete Lieder nie."""
            if int(ids[start + k]) in bewertet:
                return 0, None, None
            if g1 in fein and P[k, j1] >= fein[g1] and naehe[k, j1] >= mindest.get(g1, 1.0):
                return 1, None, None
            a, gesehen = g1, set()   # grob: nächstes sicheres Obergenre über dem wahrscheinlichsten Genre
            while a is not None and a not in gesehen:
                gesehen.add(a)
                if a in grob and a in p_ober and p_ober[a][k] >= grob[a] and n_ober[a][k] >= ober_mindest.get(a, 1.0):
                    return 0, a, float(p_ober[a][k])
                a = eltern.get(a)
            return 0, None, None

        for k in range(P.shape[0]):
            j1, j2 = top[k, 0], top[k, 1]
            g1, g2 = klassen[j1], klassen[j2]
            p1, p2 = float(P[k, j1]), float(P[k, j2])
            lid = int(ids[start + k])
            auto, ober_id, ober_p = einordnen(k, j1, g1, schwellen, nur_ober)
            pruef_id = None
            if pruef_fein or pruef_ober:  # was ein widerlegtes Genre bekäme: Kandidat für weitere Stichproben
                a2, o2, _ = einordnen(k, j1, g1, alle_fein, alle_ober)
                if a2 and g1 in pruef_fein:
                    pruef_id = g1
                elif not a2 and o2 in pruef_ober:
                    pruef_id = o2
            zeilen.append((lid, g1, p1, g2, p2, auto, jetzt, ober_id, ober_p, pruef_id))

    if trocken:
        auto_n = collections.Counter(z[1] for z in zeilen if z[5])
        ober_n = collections.Counter(z[7] for z in zeilen if z[7] is not None)
        pruef_n = collections.Counter(z[9] for z in zeilen if z[9] is not None)
        return {"auto": dict(auto_n), "ober": dict(ober_n), "pruef": dict(pruef_n), "ober_schwellen": ober,
                "schwellen": schwellen, "belegt": belegt}

    with _sperre:
        if not _schreiben_beginnen(con, genre_stand):
            return False
        con.execute("DELETE FROM vorhersagen")
        con.execute("DELETE FROM modell_stand")
        con.executemany(
            "INSERT INTO modell_stand(genre_id, n, schwelle, trefferquote, n_sicher, stichprobe_ok, stichprobe_n, ts, "
            "quote_roh, mindest, belegt, ober_schwelle, ober_quote, ober_n, ober_mindest, widerlegt, pruef_schwelle, "
            "pruef_ober_schwelle) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(g, zaehl.get(g, 0), schwellen.get(g), quoten.get(g), n_sicher.get(g), *stich.get(g, (0, 0)), jetzt,
              roh.get(g), mindest.get(g), belegt.get(g, 0), *(ober[g] if g in ober else (None, None, None)),
              ober_mindest.get(g) if g in ober or g in pruef_ober else None,
              int(g in pruef_fein or g in pruef_ober), pruef_fein.get(g), pruef_ober[g][0] if g in pruef_ober else None)
             for g in set(zaehl) | set(ober) | set(pruef_fein) | set(pruef_ober)])
        if verwechselt is not None:
            con.execute("DELETE FROM verwechslungen")
            con.executemany("INSERT INTO verwechslungen VALUES (?,?,?,?)", verwechselt)
        con.executemany("INSERT INTO vorhersagen(lied_id, genre_id, p, zweit_id, p2, auto, ts, ober_id, ober_p, pruef_id) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)", zeilen)
        if cv_faellig:
            _info(con, "cv_bewertungen", len(et))
            _info(con, "cv_reif", json.dumps(reif))
            _info(con, "merkmale", merkmale)
            _info(con, "cv_sprache", spr_bewertet)
        con.commit()
    return True


def verwechslungen_finden(y, vorhergesagt, min_n=5, ab_anteil=0.2):
    """Genre-Paare, bei denen ≥ 20 % der Lieder beim jeweils anderen landen.

    Hinweis zum Zusammenlegen: Was das Modell am Klang nicht trennen kann,
    wird nie automatisch sortiert und kostet dauerhaft Handarbeit.
    """
    klassen = np.unique(y)
    n = {g: int((y == g).sum()) for g in klassen}
    erg = []
    for i, a in enumerate(klassen):
        for b in klassen[i + 1:]:
            if n[a] < min_n or n[b] < min_n:
                continue
            falsch = int(((y == a) & (vorhergesagt == b)).sum() + ((y == b) & (vorhergesagt == a)).sum())
            anteil = falsch / (n[a] + n[b])
            if anteil >= ab_anteil:
                erg.append((int(a), int(b), float(anteil), n[a] + n[b]))
    return erg


def naechste(con, anzahl=3, ausschliessen=()):
    """Backlog-Mischung, alles direkt aus der Datenbank:

    - Stichprobe (~15 %): ein automatisch zugeordnetes Lied, Genre zuerst – bevorzugt, wo eine
      richtige Antwort die meisten Lieder freischaltet (stichproben_gewichte). Ist ein großes Genre
      nur noch wenige Antworten vom Beleg entfernt, kommen nur noch Stichproben (Vorrang).
    - Klanggruppe: noch unbenannte Gruppen, größte zuerst.
    - Verwechsler (~25 % der übrigen): Lied, bei dem das Modell genau zwischen einem oft verwechselten
      Paar schwankt (z. B. House/Tech House), Paar gewichtet nach Anteil × Bewertungen. Jede Antwort
      schärft so genau die Grenze, an der das Modell scheitert.
    - Mehr Beispiele: Genres, die laut Modell häufig sind, aber < 30 Bewertungen haben.
    - Unsicher: kleinster Abstand zwischen den zwei besten Genres.

    Von Duplikaten kommt nur die beste Fassung (die behaltene Datei der Gruppe), und keine Kopie eines
    Liedes, das schon bewertet ist oder gerade in der Schlange liegt (ausschliessen).

    Fokus (modell_info „fokus“ = Genre-ID): nur Lieder, die das Modell in diesem Genre oder seinen
    Unter-Genres sieht; keine Klanggruppen, mehr Verwechsler, „mehr Beispiele“ auch für kleine Stile.
    """
    fokus = fokus_lesen(con)
    im_fokus = _baum(con)[0][fokus] if fokus is not None else None
    def drin(*gs):
        return im_fokus is None or all(g in im_fokus for g in gs)
    nur_fokus = f" AND v.genre_id IN ({','.join(map(str, im_fokus))})" if im_fokus is not None else ""
    bewertet = {r[0] for r in con.execute("SELECT DISTINCT lied_id FROM etiketten")}
    # „Später“ heißt später: nach ein paar Tagen darf ein übersprungenes Lied wiederkommen
    uebersprungen = {r[0] for r in con.execute("SELECT lied_id FROM uebersprungen WHERE ts > ?",
                                                (time.time() - config.SPAETER_TAGE * 86400,))}
    tabu = bewertet | uebersprungen | set(ausschliessen)
    familie, nur_kopie = duplikate.kopien(con)
    familie_tabu = {familie[l] for l in bewertet | set(ausschliessen) if l in familie}
    analysiert = "JOIN lieder l ON l.id={}.lied_id AND l.status='analysiert'"

    groesse = dict(con.execute(f"SELECT k.gruppe, COUNT(*) FROM klanggruppen k {analysiert.format('k')} "
                               "GROUP BY k.gruppe").fetchall())
    benannt = {r[0] for r in con.execute(
        "SELECT DISTINCT k.gruppe FROM klanggruppen k JOIN etiketten e ON e.lied_id=k.lied_id")}
    offen_gruppen = sorted((g for g in groesse if g not in benannt), key=lambda g: -groesse[g]) if fokus is None else []
    gruppe_von = {}
    pruef_von = {}   # Stichprobe: welches Genre (fein oder Obergenre) geprüft wird
    hat_modell = con.execute("SELECT 1 FROM vorhersagen LIMIT 1").fetchone() is not None
    stich = _stichproben(con)
    # geprüft wird jede Stufe: fein (auto) und grob (Obergenre)
    # widerlegte Genres (pruef_id) ebenso, sonst könnten sie sich nie erholen
    wartend = dict(con.execute("SELECT g, COUNT(*) FROM (SELECT genre_id g FROM vorhersagen WHERE auto=1 "
                               "UNION ALL SELECT ober_id FROM vorhersagen WHERE ober_id IS NOT NULL "
                               "UNION ALL SELECT pruef_id FROM vorhersagen WHERE pruef_id IS NOT NULL) GROUP BY g"))
    auto_genres = [(g, w) for g, w in stichproben_gewichte(wartend, stich) if drin(g)]
    # Vorrang: ein großes Genre ist nur noch wenige richtige Antworten vom Beleg entfernt → nur Stichproben
    # für diese Genres, bis sie belegt sind (oder eine falsche Antwort den Bedarf über die Grenze treibt)
    vorrang = [(g, w) for g, w in auto_genres if wartend[g] >= config.STICHPROBE_VORRANG_LIEDER
               and 0 < (stichproben_fehlen(*stich.get(g, (0, 0))) or 0) <= config.STICHPROBE_VORRANG_FEHLT]
    n_bew = dict(con.execute("SELECT genre_id, COUNT(*) FROM etiketten WHERE haupt=1 GROUP BY genre_id").fetchall())
    haeufig = [(g, n) for g, n in con.execute("SELECT genre_id, COUNT(*) FROM vorhersagen GROUP BY genre_id")
               if n >= (50 if fokus is None else config.FOKUS_AUFFUELLEN_AB)
               and n_bew.get(g, 0) < config.MIN_BEWERTUNGEN_AUTO and drin(g)]
    paare = [((a, b), anteil * n) for a, b, anteil, n in con.execute("SELECT a, b, anteil, n FROM verwechslungen")
             if drin(a, b)]
    verwechsler_anteil = config.VERWECHSLER_ANTEIL if fokus is None else config.FOKUS_VERWECHSLER
    zufall = random.Random(time.time())

    def frei(sql, *werte):
        for (lid,) in con.execute(sql, werte):
            if lid not in tabu and lid not in nur_kopie and familie.get(lid) not in familie_tabu:
                return lid
        return None

    def gewichtet(paare):
        gesamt = sum(w for _, w in paare)
        r = zufall.random() * gesamt
        for x, w in paare:
            r -= w
            if r <= 0:
                return x
        return paare[-1][0]

    ergebnis = []
    for _ in range(anzahl):
        wahl = grund = None
        wurf = zufall.random()
        if vorrang or (auto_genres and wurf < config.STICHPROBE_ANTEIL):
            g = gewichtet(vorrang or auto_genres)
            wahl = frei(f"SELECT v.lied_id FROM vorhersagen v {analysiert.format('v')} WHERE (v.auto=1 AND v.genre_id=?) "
                        "OR v.ober_id=? OR v.pruef_id=? ORDER BY random() LIMIT 150", g, g, g)
            grund = "stichprobe"
            pruef_von[wahl] = g
        if wahl is None and offen_gruppen and (not hat_modell or wurf < 0.5):
            while offen_gruppen and wahl is None:
                g = offen_gruppen.pop(0)
                wahl = frei(f"SELECT k.lied_id FROM klanggruppen k {analysiert.format('k')} WHERE k.gruppe=? "
                            "ORDER BY k.abstand LIMIT 90", g)
                gruppe_von[wahl] = groesse[g]
            grund = "gruppe"
        if wahl is None and paare and zufall.random() < verwechsler_anteil:
            a, b = gewichtet(paare)
            wahl = frei(f"SELECT v.lied_id FROM vorhersagen v {analysiert.format('v')} WHERE v.auto=0 "
                        "AND ((v.genre_id=? AND v.zweit_id=?) OR (v.genre_id=? AND v.zweit_id=?)) "
                        "ORDER BY v.p - coalesce(v.p2, 0) LIMIT 300", a, b, b, a)
            grund = "verwechsler"
        if wahl is None and haeufig and wurf < (0.75 if fokus is None else 0.9):
            g = gewichtet(haeufig)
            wahl = frei(f"SELECT v.lied_id FROM vorhersagen v {analysiert.format('v')} WHERE v.genre_id=? AND v.auto=0 "
                        "ORDER BY random() LIMIT 150", g)
            grund = "auffuellen"
        if wahl is None and hat_modell:
            wahl = frei(f"SELECT v.lied_id FROM vorhersagen v {analysiert.format('v')} WHERE v.auto=0{nur_fokus} "
                        "ORDER BY v.p - coalesce(v.p2, 0) LIMIT 1500")
            grund = "unsicher"
        if wahl is None and fokus is not None:
            wahl = frei(f"SELECT v.lied_id FROM vorhersagen v {analysiert.format('v')} WHERE 1{nur_fokus} "
                        "ORDER BY random() LIMIT 600")
            grund = "zufall"
        if wahl is None and fokus is None:
            wahl = frei("SELECT id FROM lieder WHERE status='analysiert' AND emb IS NOT NULL ORDER BY random() LIMIT 600")
            grund = "zufall"
        if wahl is None:
            break
        tabu.add(wahl)
        if wahl in familie:
            familie_tabu.add(familie[wahl])
        ergebnis.append({"lied_id": wahl, "grund": grund, "gruppe": gruppe_von.get(wahl, 0),
                         **({"pruef_id": pruef_von[wahl]} if grund == "stichprobe" and wahl in pruef_von else {})})
    return ergebnis


def fokus_lesen(con):
    """Genre-ID des Bewertungs-Fokus oder None (alle). Ein gelöschtes Genre gilt als kein Fokus."""
    f = _info(con, "fokus")
    if not f:
        return None
    return int(f) if con.execute("SELECT 1 FROM genres WHERE id=?", (int(f),)).fetchone() else None


def fokus_setzen(con, genre_id):
    with con:
        if genre_id is None:
            con.execute("DELETE FROM modell_info WHERE schluessel='fokus'")
        else:
            _info(con, "fokus", int(genre_id))


def cluster_berechnen(con=None, pro_gruppe=150):
    """Klanggruppen (MiniBatch-k-Means) für alle analysierten Lieder; eigener Prozess,
    damit die App ihren Speicher nicht dafür braucht."""
    from sklearn.cluster import MiniBatchKMeans
    con = con or db.verbinden()
    b = Bestand()
    b.aktualisieren(con, erzwingen=True)
    n = len(b.ids)
    if n < 20:
        return 0
    k = int(np.clip(n / pro_gruppe, 2, 800))
    km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3, init_size=min(n, 3 * k + 3000))
    gruppen = np.empty(n, dtype=np.int64)
    # erst lernen (in Teilstücken), dann zuordnen: nie die ganze Matrix als float32
    km.fit(b.X[:20000].astype(np.float32))
    for s in range(20000, n, 20000):
        km.partial_fit(b.X[s:s + 20000].astype(np.float32))
    abstand = np.empty(n, dtype=np.float32)
    for s in range(0, n, 20000):
        teil = b.X[s:s + 20000].astype(np.float32)
        gruppen[s:s + 20000] = km.predict(teil)
        abstand[s:s + 20000] = ((teil - km.cluster_centers_[gruppen[s:s + 20000]]) ** 2).sum(1)
    with con:
        con.execute("DELETE FROM klanggruppen")
        con.executemany("INSERT INTO klanggruppen VALUES (?,?,?)",
                        zip(map(int, b.ids), map(int, gruppen), map(float, abstand)))
    return k


_auftrag = threading.Event()
_trainer = None


def sperre():
    """Sperre innerhalb des App-Prozesses (Genre-Änderungen untereinander). Gegen das Training
    im eigenen Prozess schützt der Zähler modell_info.genre_stand (siehe trainieren)."""
    return _sperre


def trainieren_im_hintergrund():
    """Neu trainieren, ohne dass sich Läufe stapeln: viele schnelle Bewertungen = ein Lauf danach."""
    global _trainer
    _auftrag.set()
    if _trainer is None or not _trainer.is_alive():
        def schleife():
            while _auftrag.wait():  # läuft dauerhaft; ein Auftrag geht so nie verloren
                _auftrag.clear()
                # eigener Prozess: Rechnen blockiert die App nicht (GIL), Absturz trifft sie nicht
                r = subprocess.run([sys.executable, "-m", "musiksort.lernen", "trainieren"],
                                   capture_output=True, text=True, preexec_fn=lambda: os.nice(5))
                if r.returncode == 3:  # Genres zwischendurch geändert: gleich nochmal
                    _auftrag.set()
                elif r.returncode:
                    print("Training fehlgeschlagen:", r.stderr[-500:], flush=True)
                    if "database is locked" in r.stderr:  # DB kurz gesperrt: gleich nochmal
                        time.sleep(10)
                        _auftrag.set()
        _trainer = threading.Thread(target=schleife, daemon=True)
        _trainer.start()


if __name__ == "__main__":
    if sys.argv[1:] == ["cluster"]:
        t = time.time()
        print(f"{cluster_berechnen()} Klanggruppen in {time.time() - t:.0f} s", flush=True)
    elif sys.argv[1:2] == ["trainieren"]:
        if not trainieren(cv_erzwingen="--cv" in sys.argv):
            print("Genres wurden währenddessen geändert: Ergebnis verworfen")
            sys.exit(3)
        print("trainiert")
    else:
        sys.exit(__doc__)
