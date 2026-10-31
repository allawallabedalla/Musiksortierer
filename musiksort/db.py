"""SQLite-Datenbank: Lieder, Genres, Bewertungen, Vorhersagen, Duplikate."""
import sqlite3
import time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS lieder (
    id INTEGER PRIMARY KEY,
    pfad TEXT UNIQUE NOT NULL,          -- relativ zu MEDIA_ROOT
    groesse INTEGER, mtime REAL,
    status TEXT NOT NULL DEFAULT 'neu', -- neu | analysiert | fehler | fehlt (nicht mehr auf dem NAS) | ausgeschlossen
    fehler TEXT,
    codec TEXT, bitrate INTEGER, samplerate INTEGER, bittiefe INTEGER,
    kanaele INTEGER, dauer REAL, verlustfrei INTEGER,
    interpret TEXT, titel TEXT, album TEXT, tag_genre TEXT,
    bpm REAL,
    tempo_stand TEXT, bpm_in_datei INTEGER,
    snippet_start REAL, snippet_quelle TEXT,  -- energie | 1:45 | anteil
    emb BLOB,                                  -- float16[1280], Hauptteil
    emb_std BLOB,                              -- float16[1280], Streuung über die Zeitfenster (Zeitverlauf)
    discogs TEXT,                              -- JSON Top-5 Discogs-Stile
    fp BLOB,                                   -- int32[] Chromaprint roh
    analysiert REAL, analyse_s REAL,
    versuche INTEGER NOT NULL DEFAULT 0,
    sha256 TEXT, sha_stand TEXT, fp_ende BLOB, acoustid TEXT,  -- für die automatische Duplikat-Entscheidung
    stille_anfang REAL, stille_ende REAL, fp_ende2 BLOB,        -- Ende ohne Stille (klang.py)
    grenzfrequenz REAL, kante REAL,                             -- falsche FLACs erkennen (klang.py)
    stern INTEGER NOT NULL DEFAULT 0, stern_ts REAL,            -- Favorit (Stern in der App)
    stern_in_datei INTEGER                                      -- Stand, der zuletzt in die Datei geschrieben wurde (tags.py)
);
CREATE INDEX IF NOT EXISTS lieder_status ON lieder(status);

CREATE TABLE IF NOT EXISTS genres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- nie wiederverwendet: Stichproben/Etiketten hängen an der ID
    name TEXT NOT NULL,
    eltern_id INTEGER REFERENCES genres(id),
    erstellt REAL,
    UNIQUE(name, eltern_id)
);

-- Deine Antworten. haupt=1: Hauptgenre (Ordner), haupt=0: Zusatz-Tag
CREATE TABLE IF NOT EXISTS etiketten (
    lied_id INTEGER NOT NULL REFERENCES lieder(id),
    genre_id INTEGER NOT NULL REFERENCES genres(id),
    haupt INTEGER NOT NULL DEFAULT 1,
    quelle TEXT NOT NULL,       -- ich | stichprobe (Abgleich mit dem Modell: Tabelle stichproben)
    ts REAL NOT NULL,
    PRIMARY KEY (lied_id, genre_id)
);

-- Startmodell (startmodell.py): fremde, anonyme Beispiele; zählen nur fürs Modell, nie für Freigaben
CREATE TABLE IF NOT EXISTS startbeispiele (
    id INTEGER PRIMARY KEY,
    genre_id INTEGER NOT NULL REFERENCES genres(id),
    emb BLOB NOT NULL,          -- float16 × 1280, normiert
    dauer REAL,
    sprache BLOB                -- float16 × sprache.BREITE, ohne Gewicht
);

CREATE TABLE IF NOT EXISTS uebersprungen (
    lied_id INTEGER PRIMARY KEY REFERENCES lieder(id),
    ts REAL NOT NULL, anzahl INTEGER NOT NULL DEFAULT 1
);

-- Letzter Stand des Lernmodells pro Lied
CREATE TABLE IF NOT EXISTS vorhersagen (
    lied_id INTEGER PRIMARY KEY REFERENCES lieder(id),
    genre_id INTEGER, p REAL, zweit_id INTEGER, p2 REAL,
    auto INTEGER NOT NULL DEFAULT 0,   -- 1 = sicher genug für automatisch
    ts REAL,
    ober_id INTEGER, ober_p REAL,      -- grob einsortiert: sicheres Obergenre, wenn das Unter-Genre unsicher ist
    pruef_id INTEGER                   -- Genre (fein/grob), das die Stichproben widerlegt haben: wird weiter geprüft
);

CREATE TABLE IF NOT EXISTS modell_stand (
    genre_id INTEGER PRIMARY KEY,
    n INTEGER, schwelle REAL, trefferquote REAL, n_sicher INTEGER,
    stichprobe_ok INTEGER, stichprobe_n INTEGER, ts REAL,
    quote_roh REAL, mindest REAL, belegt INTEGER NOT NULL DEFAULT 0,
    ober_schwelle REAL, ober_quote REAL, ober_n INTEGER, ober_mindest REAL,  -- als Obergenre (ganzer Teilbaum)
    -- von Stichproben widerlegt: Schwellen ruhen hier, damit weiter geprüft wird (erholt sich das Genre: wieder aktiv)
    widerlegt INTEGER NOT NULL DEFAULT 0, pruef_schwelle REAL, pruef_ober_schwelle REAL
);

-- Klanggruppen (k-Means auf den Steckbriefen), berechnet von `python -m musiksort.lernen cluster`
CREATE TABLE IF NOT EXISTS klanggruppen (
    lied_id INTEGER PRIMARY KEY, gruppe INTEGER NOT NULL, abstand REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS klanggruppen_gruppe ON klanggruppen(gruppe, abstand);

-- Stichproben: was das Modell automatisch einsortiert hätte, und was du gewählt hast
CREATE TABLE IF NOT EXISTS stichproben (
    id INTEGER PRIMARY KEY, lied_id INTEGER NOT NULL, vorhergesagt INTEGER NOT NULL,
    gewaehlt INTEGER NOT NULL, ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS modell_info (schluessel TEXT PRIMARY KEY, wert TEXT);

-- Zurück-Knopf: Zustand vor einer Bewertung/einem „Später“, um ihn wiederherzustellen
CREATE TABLE IF NOT EXISTS rueckgaengig (
    id INTEGER PRIMARY KEY, lied_id INTEGER NOT NULL, art TEXT NOT NULL,  -- bewertet | uebersprungen
    vorher TEXT NOT NULL, stichprobe_id INTEGER, ts REAL NOT NULL
);

-- Genre-Paare, die das Modell oft verwechselt (Kreuzvalidierung): Kandidaten fürs Zusammenlegen
CREATE TABLE IF NOT EXISTS verwechslungen (
    a INTEGER NOT NULL, b INTEGER NOT NULL,
    anteil REAL NOT NULL,   -- Anteil der Lieder beider Genres, die beim jeweils anderen landeten
    n INTEGER NOT NULL,     -- Bewertungen beider Genres zusammen
    PRIMARY KEY (a, b)
);

CREATE TABLE IF NOT EXISTS dup_gruppen (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- nie wiederverwendet: eine Freigabe trifft immer genau ihre Gruppe
    status TEXT NOT NULL DEFAULT 'offen',  -- offen | freigegeben | verworfen | ersetzt (ungültig/neu berechnet) | erledigt (ausgeführt, Papierkorb)
    sicher INTEGER NOT NULL DEFAULT 1,     -- 0 = Zweifel: Edit, Remaster, Live?
    erstellt REAL,
    entschieden_von TEXT                   -- ich | auto:<stufe> (zum gezielten Zurücknehmen)
);
CREATE TABLE IF NOT EXISTS dup_mitglieder (
    gruppe_id INTEGER NOT NULL REFERENCES dup_gruppen(id),
    lied_id INTEGER NOT NULL REFERENCES lieder(id),
    behalten INTEGER NOT NULL DEFAULT 0,
    abweichung REAL,                         -- Bitfehlerrate zum Behaltenen
    PRIMARY KEY (gruppe_id, lied_id)
);

-- Jede Dateibewegung und jede folgenreiche Entscheidung, damit alles nachvollziehbar/rückgängig ist
CREATE TABLE IF NOT EXISTS protokoll (
    id INTEGER PRIMARY KEY, ts REAL, aktion TEXT, von TEXT, nach TEXT, lied_id INTEGER
);
"""


def verbinden(pfad=None, schema=True, threads=False) -> sqlite3.Connection:
    pfad = pfad or config.DB_PFAD
    pfad.parent.mkdir(parents=True, exist_ok=True)
    # threads=True: die Web-App schließt Verbindungen nach der Anfrage in einem anderen Thread
    con = sqlite3.connect(pfad, timeout=30, check_same_thread=not threads)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    if schema:
        con.executescript(SCHEMA)
        _nachruesten(con)
    return con


def _autoincrement(con, tabelle):
    """Ältere DBs: Tabelle mit AUTOINCREMENT neu aufbauen (Inhalt und IDs bleiben)."""
    sql = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tabelle,)).fetchone()[0]
    if "AUTOINCREMENT" in sql.upper():
        return
    con.execute("BEGIN IMMEDIATE")
    try:
        sql = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tabelle,)).fetchone()[0]
        if "AUTOINCREMENT" not in sql.upper():  # inzwischen von einem anderen Prozess erledigt?
            neu = sql.replace("id INTEGER PRIMARY KEY", "id INTEGER PRIMARY KEY AUTOINCREMENT", 1)
            neu = neu.replace(f"CREATE TABLE {tabelle}", f"CREATE TABLE {tabelle}_neu", 1)
            con.execute(neu)
            con.execute(f"INSERT INTO {tabelle}_neu SELECT * FROM {tabelle}")
            con.execute(f"DROP TABLE {tabelle}")
            con.execute(f"ALTER TABLE {tabelle}_neu RENAME TO {tabelle}")
        con.commit()
    except Exception:
        con.rollback()
        raise


def _nachruesten(con):
    """Spalten, die nach dem ersten Anlegen dazukamen, in bestehende DBs einfügen."""
    for tabelle in ("genres", "dup_gruppen"):
        _autoincrement(con, tabelle)
    spalten = {r[1] for r in con.execute("PRAGMA table_info(lieder)")}
    for spalte, typ in (("versuche", "INTEGER NOT NULL DEFAULT 0"), ("sha256", "TEXT"), ("sha_stand", "TEXT"),
                        ("fp_ende", "BLOB"), ("acoustid", "TEXT"),
                        ("stille_anfang", "REAL"), ("stille_ende", "REAL"), ("fp_ende2", "BLOB"),
                        ("grenzfrequenz", "REAL"), ("kante", "REAL"), ("stern", "INTEGER NOT NULL DEFAULT 0"),
                        ("stern_ts", "REAL"), ("stern_in_datei", "INTEGER"),
                        ("sprache", "TEXT"), ("sprache_p", "REAL"), ("sprache_probs", "TEXT"), ("sprache_ts", "REAL"),
                        ("sprache_in_datei", "TEXT"), ("wellenform", "BLOB"), ("tempo_stand", "TEXT"), ("bpm_in_datei", "INTEGER"),
                        ("emb_std", "BLOB")):
        if spalte not in spalten:
            con.execute(f"ALTER TABLE lieder ADD COLUMN {spalte} {typ}")
    spalten = {r[1] for r in con.execute("PRAGMA table_info(modell_stand)")}
    for spalte, typ in (("quote_roh", "REAL"), ("mindest", "REAL"), ("belegt", "INTEGER NOT NULL DEFAULT 0")):
        if spalte not in spalten:
            con.execute(f"ALTER TABLE modell_stand ADD COLUMN {spalte} {typ}")
    for tabelle, neu in (("vorhersagen", (("ober_id", "INTEGER"), ("ober_p", "REAL"), ("pruef_id", "INTEGER"))),
                         ("modell_stand", (("ober_schwelle", "REAL"), ("ober_quote", "REAL"), ("ober_n", "INTEGER"),
                                           ("ober_mindest", "REAL"), ("widerlegt", "INTEGER NOT NULL DEFAULT 0"),
                                           ("pruef_schwelle", "REAL"), ("pruef_ober_schwelle", "REAL")))):
        spalten = {r[1] for r in con.execute(f"PRAGMA table_info({tabelle})")}
        for spalte, typ in neu:
            if spalte not in spalten:
                con.execute(f"ALTER TABLE {tabelle} ADD COLUMN {spalte} {typ}")
    spalten = {r[1] for r in con.execute("PRAGMA table_info(dup_gruppen)")}
    for spalte, typ in (("sicher", "INTEGER NOT NULL DEFAULT 1"), ("entschieden_von", "TEXT")):
        if spalte not in spalten:
            con.execute(f"ALTER TABLE dup_gruppen ADD COLUMN {spalte} {typ}")
    con.commit()


def jetzt() -> float:
    return time.time()
