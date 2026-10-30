"""Zentrale Einstellungen. Alles über Umgebungsvariablen überschreibbar."""
import json
import os
from pathlib import Path

PROJEKT = Path(__file__).resolve().parent.parent

# Quelle: Musik-Ordner, z. B. eine NAS-Freigabe (nur lesend genügt bis zum Sortierlauf)
MEDIA_ROOT = Path(os.environ.get("MUSIKSORT_MEDIA", "/mnt/media"))
MOUNT_PRUEFEN = os.environ.get("MUSIKSORT_MOUNT_PRUEFEN", "1") == "1"  # Tests: lokaler Ordner statt NAS


AUDIO_ENDUNGEN = {
    ".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus",
    ".aif", ".aiff", ".wma", ".ape", ".wv", ".alac", ".mp4",
}
VERLUSTFREI = {"flac", "alac", "wav", "aiff", "ape", "wavpack"}

DATEN = Path(os.environ.get("MUSIKSORT_DATEN", Path.home() / ".local/share/musiksortierer"))
CACHE = Path(os.environ.get("MUSIKSORT_CACHE", Path.home() / ".cache/musiksortierer"))
DB_PFAD = DATEN / "musik.db"

# Ordner der eigenen Sammlung stehen nicht im Code, sondern in einer lokalen Datei (siehe README):
# {"ausgeschlossen": [...], "ausgeschlossene_pfade": [...], "ordner_rang": {...},
#  "duplikat_ordner": [...], "backup_ordner": [...]}
EINSTELLUNGEN = Path(os.environ.get("MUSIKSORT_EINSTELLUNGEN", DATEN / "einstellungen.json"))
try:
    _lokal = json.loads(EINSTELLUNGEN.read_text())
except FileNotFoundError:
    _lokal = None
EINSTELLUNGEN_DA = _lokal is not None
_lokal = _lokal or {}

# Oberste Ordner, die keine Musik enthalten (Systemordner der NAS sind immer dabei)
AUSGESCHLOSSEN = {"#recycle", "@eaDir", "#snapshot", ".AppleDouble",
                  "_papierkorb",  # PAPIERKORB: vom Sortierlauf aussortierte Dateien (papierkorb.py)
                  } | set(_lokal.get("ausgeschlossen", []))
# Tiefer liegende Ordner ohne Musik (relativ zu MEDIA_ROOT)
AUSGESCHLOSSENE_PFADE = set(_lokal.get("ausgeschlossene_pfade", []))
# Duplikate: bei gleicher Qualität und gleichen Tags lieber aus diesen obersten Ordnern behalten (höher = lieber)
ORDNER_RANG = dict(_lokal.get("ordner_rang", {}))
# Oberste Ordner, die bekanntermaßen Doppel bzw. alte Sicherungen enthalten (Merkmale der Duplikat-Automatik)
DUPLIKAT_ORDNER = set(_lokal.get("duplikat_ordner", []))
BACKUP_ORDNER = set(_lokal.get("backup_ordner", []))
SNIPPETS = CACHE / "snippets"

MODELLE = PROJEKT / "modelle"
EFFNET = MODELLE / "discogs-effnet-bs64-1.pb"
DISCOGS400 = MODELLE / "genre_discogs400-discogs-effnet-1.pb"
DISCOGS400_JSON = MODELLE / "genre_discogs400-discogs-effnet-1.json"
TEMPO = MODELLE / "deeptemp-k16-3.pb"   # TempoCNN: 91 % genau gegen BPM-Tags (Percival 17 %, n=208, 30.10.2026)
TEMPO_STAND = TEMPO.stem              # steht in lieder.tempo_stand: welches Modell das BPM gemessen hat
TEMPO_SR = 11025
WHISPER = MODELLE / "whisper-tiny"   # Spracherkennung (sprache.py), faster-whisper/CTranslate2

SR = 16000                 # Abtastrate für Modell und Fingerabdruck
MAX_ANALYSE_S = 15 * 60    # lange Mixe nur die ersten 15 min analysieren
FP_SEKUNDEN = 120          # Fingerabdruck über die ersten 2 min
SNIPPET_S = 30
SNIPPET_RUECKFALL = 105    # 1:45
SNIPPET_CACHE_MB = int(os.environ.get("MUSIKSORT_CACHE_MB", "3000"))  # danach werden die ältesten gelöscht

# Automatisch einsortieren erst, wenn die geschätzte Trefferquote darüber liegt
ZIEL_TREFFERQUOTE = float(os.environ.get("MUSIKSORT_ZIEL", "0.95"))
# Vorsichtige Untergrenze (Wilson, einseitig 90 %): „≥ 95 %“ braucht damit ≥ 32 fehlerfreie Fälle.
# z=1 ließ in Simulationen Genres mit echten 90 % in 13–16 % der Läufe durch.
WILSON_Z = 1.28
MIN_BEWERTUNGEN_AUTO = 30  # pro Genre; darunter lässt sich ≥ 95 % ohnehin nicht vorsichtig belegen
SPAETER_TAGE = 3           # übersprungene Lieder kommen danach wieder
STICHPROBE_ANTEIL = 0.15   # Anteil der Karten, die eine automatische Zuordnung prüfen
VERWECHSLER_ANTEIL = 0.25  # Anteil der übrigen Karten für Lieder genau zwischen zwei oft verwechselten Genres
VERWECHSLUNG_MERKEN = 0.05  # Paare ab diesem Verwechslungs-Anteil gezielt abfragen …
VERWECHSLUNG_HINWEIS = 0.2  # … ab diesem in der App zum Zusammenlegen vorschlagen
STICHPROBE_MIN = 10        # so viele Stichproben je Genre, bevor sie zählen
STICHPROBE_ZIEL = 0.90     # positiver Beleg fürs Verschieben: Untergrenze der Stichproben-Quote
STICHPROBE_NACHPRUEFEN = 200  # belegte/aussichtslose Genres: Gewicht wie bei so vielen fehlenden Stichproben
STICHPROBE_VORRANG_LIEDER = 1000  # Genre so groß und …
STICHPROBE_VORRANG_FEHLT = 10     # … höchstens so viele richtige Antworten vom Beleg: nur noch Stichproben
AUSREISSER_QUANTIL = 0.10  # weiter weg vom Genre-Zentrum als 90 % der eigenen Beispiele: nicht automatisch
LAENGE_GEWICHT = 1.0       # Gewicht der Längen-Merkmale neben dem Klang (Mixtapes erkennen)
START_GEWICHT = 0.5        # Gewicht eines Startmodell-Beispiels gegenüber einer eigenen Bewertung
SPRACHE_GEWICHT = 1.0      # Gewicht der Gesangssprache (Deutschrap vs. US-Rap klingt gleich, Sprache trennt)
SPRACHE_ZUERST = ("Hip Hop",)  # Lieder, die das Modell hier sieht, bekommen ihre Sprache zuerst gemessen
SPRACHE_DATEI_P = 0.7      # erst ab dieser Sicherheit wird die Sprache in die Datei geschrieben
PAPIERKORB = "_papierkorb"   # Ordner in Media für aussortierte Dateien (papierkorb.py)
PAPIERKORB_TAGE = 30        # erst danach endgültig löschen
DUP_AUTO_D_MIN = 200       # so viele eigene Duplikat-Entscheidungen, bevor Stufe D (Lernen) mitentscheidet
DUP_AUTO_D_ZIEL = 0.98     # vorsichtige Trefferquote, ab der Stufe D entscheidet
MODELL_C = 8.0            # Regularisierung des Genre-Modells (C=2 → 8: +2,4 Punkte Treffer, Messung 30.10.2026)
CV_ALLE = 25               # teure Kreuzvalidierung nur alle so vielen neuen Bewertungen

PORT = int(os.environ.get("MUSIKSORT_PORT", "8097"))


def ausgeschlossen(rel):
    """Liegt ein relativer Pfad (Ordner oder Datei) in einem ausgeschlossenen Bereich?"""
    teile = rel.split("/")
    return teile[0] in AUSGESCHLOSSEN or any(rel == p or rel.startswith(p + "/") for p in AUSGESCHLOSSENE_PFADE)


def medienpfad(rel):
    """Absoluter Pfad einer Datei unter MEDIA_ROOT; nie außerhalb (kein ../, keine absoluten Pfade)."""
    pfad = (MEDIA_ROOT / rel).resolve()
    wurzel = MEDIA_ROOT.resolve()
    if pfad != wurzel and wurzel not in pfad.parents:
        raise ValueError(f"Pfad außerhalb der Freigabe: {rel}")
    return pfad


def env(name, standard=None):
    """Erst Umgebung, dann ~/.env. Anführungszeichen nur entfernen, wenn sie den
    Wert umschließen: ein Passwort darf selbst auf ' oder " enden."""
    if name in os.environ:
        return os.environ[name]
    datei = Path.home() / ".env"
    if datei.exists():
        for zeile in datei.read_text().splitlines():
            if zeile.strip().startswith(name + "="):
                wert = zeile.split("=", 1)[1].strip()
                if len(wert) >= 2 and wert[0] == wert[-1] and wert[0] in "\"'":
                    wert = wert[1:-1]
                return wert
    return standard
