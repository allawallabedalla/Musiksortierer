# Musiksortierer

Eine große, gewachsene Musiksammlung (getestet mit rund 100.000 Dateien aus mehreren alten Backups) nach **eigenen** Genres
ordnen und Duplikate loswerden – ohne dass die Maschine rät, wo sie sich nicht sicher ist.

Man bewertet Lieder in einer Web-App (am Handy gut bedienbar): kurz reinhören, Genre antippen, weiter.
Ein Modell lernt daraus und sortiert ein Genre erst dann selbst, wenn es nachweislich trifft:
vorsichtige Trefferquote ≥ 95 % in der Kreuzvalidierung **und** bestandene Stichproben, die man beim Bewerten
blind gegenprüft. Bis dahin wird nichts verschoben.

## Was es kann

- **Klang-Analyse** jedes Liedes mit [Essentia](https://essentia.upf.edu) (Discogs-EffNet-Steckbrief, BPM per TempoCNN, Länge)
- **Gesangssprache** mit [Whisper](https://github.com/openai/whisper) (nur die Spracherkennung, Modell „tiny“) –
  trennt z. B. Deutschrap von US-Rap, die gleich klingen
- **Bewerten-App**: Ausschnitt anhören, Genre wählen, Zurück-Knopf, Favoriten-Stern, Rückmeldung bei Stichproben
- **Eigener Genre-Baum** mit Obergenres: sind Unter-Genres noch nicht unterscheidbar, wird erst grob einsortiert
- **Aktives Lernen**: die App legt die Lieder vor, deren Bewertung am meisten bringt
- **Duplikate**: Chromaprint-Fingerabdrücke, mehrstufige Automatik (bitgleich, gleiche Aufnahme in anderer Qualität,
  [AcoustID](https://acoustid.org)-Abgleich, aus den eigenen Entscheidungen gelernt), falsche FLACs („aus MP3“) erkennen
- **Papierkorb** statt Löschen: 30 Tage, Datei für Datei zurückholbar
- **Tags in die Datei**: Favorit (POPM/RATING) und Sprache (TLAN/LANGUAGE), wenn die Freigabe beschreibbar ist

## Voraussetzungen

- Linux mit systemd (getestet: Debian 13, Python 3.13), `ffmpeg`, `libchromaprint1`
- Die Musik als Ordner, z. B. eine NAS-Freigabe (bis zum Sortierlauf genügt nur lesend)
- Etwa 4 GB RAM frei; die Analyse braucht je nach CPU ~1–3 s pro Lied

## Einrichten

```bash
git clone https://github.com/allawallabedalla/Musiksortierer.git ~/musiksortierer
cd ~/musiksortierer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Modelle** (nicht im Repo, siehe Lizenzen) nach `modelle/` laden:

```bash
mkdir -p modelle && cd modelle
for f in feature-extractors/discogs-effnet/discogs-effnet-bs64-1 classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1; do
  curl -fLO "https://essentia.upf.edu/models/$f.pb"; curl -fLO "https://essentia.upf.edu/models/$f.json"
done
curl -fLO https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb   # Tempo (BPM)
curl -fLO https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.json
cd .. && .venv/bin/python -c "from faster_whisper import download_model; download_model('tiny', output_dir='modelle/whisper-tiny')"
```

**Eigene Ordner** in `~/.local/share/musiksortierer/einstellungen.json` (alle Angaben optional, Pfade relativ zum Musik-Ordner):

```json
{
  "ausgeschlossen": ["Videos", "Bilder"],
  "ausgeschlossene_pfade": ["Archiv/2010/Systemklaenge"],
  "ordner_rang": {"Musik": 5, "Archiv": 2, "Doppelt": 0},
  "duplikat_ordner": ["Doppelt"],
  "backup_ordner": ["Archiv"]
}
```

`ordner_rang` entscheidet bei sonst gleichen Duplikaten, welche Datei bleibt (höher = lieber).
Fehlt die Datei, sucht der Musiksortierer bei einer bestehenden Sammlung nicht weiter, statt still anders zu suchen.

**Umgebung**: `MUSIKSORT_*` für die Web-App in `~/.config/musiksortierer/web.env`, `ACOUSTID_KEY` in `~/.config/musiksortierer/acoustid.env` (die abgeschotteten Dienste sehen `~/.env` nicht), `PUSHOVER_*` in `~/.env`:

| Variable | Bedeutung | Standard |
|---|---|---|
| `MUSIKSORT_MEDIA` | Musik-Ordner | `/mnt/media` |
| `MUSIKSORT_DATEN` | Datenbank, Einstellungen | `~/.local/share/musiksortierer` |
| `MUSIKSORT_CACHE` | Ausschnitte, Zwischendateien | `~/.cache/musiksortierer` |
| `MUSIKSORT_PORT` | Port der Web-App | `8097` |
| `MUSIKSORT_SCHLUESSEL` | Zugangsschlüssel der Web-App (leer = ohne) | – |
| `MUSIKSORT_HOSTS` | erlaubte Hostnamen der Web-App | – |
| `ACOUSTID_KEY` | eigener [AcoustID-Anwendungsschlüssel](https://acoustid.org/new-application) für Duplikat-Stufe B | – |
| `PUSHOVER_TOKEN`, `PUSHOVER_USER` | Push, wenn ein Dienst scheitert (sonst nur Journal) | – |

**Dienste** (systemd, Benutzer-Units):

```bash
sh deploy/installieren.sh
```

Die Units drosseln sich selbst (`musiksort.slice`: höchstens 2 Kerne, 3,5 GB) und binden die rechenintensiven
Jobs per `taskset -c 4,5` an zwei Kerne – auf Rechnern mit weniger Kernen dort anpassen.
Danach läuft die App unter `http://<rechner>:8097`.

**Startmodell** (optional, für eine neue Sammlung): `startmodell/` enthält gut 1.400 bewertete Beispiele
des Herausgebers – nur Klang-Steckbrief, Länge, Gesangssprache und Genre, keine Dateinamen, Interpreten oder
Titel – und ein daraus trainiertes Modell. Geladen geben sie vom ersten Lied an Genre-Vorschläge (mit halbem
Gewicht neben deinen eigenen Bewertungen). Selbst sortiert wird trotzdem erst, was deine eigenen Bewertungen und
Stichproben belegen. Die Genres kommen dabei mit (lassen sich in der App umbenennen, verschieben, zusammenführen).

```bash
.venv/bin/python -m musiksort.startmodell laden       # entfernen: … startmodell entfernen
```

## Befehle

```bash
.venv/bin/python -m musiksort.scan suchen          # Dateien finden
.venv/bin/python -m musiksort.scan analysieren     # Klang-Steckbriefe
.venv/bin/python -m musiksort.scan tempo           # BPM des Bestands mit neuem Tempo-Modell nachmessen
.venv/bin/python -m musiksort.sprache              # Gesangssprache messen
.venv/bin/python -m musiksort.dupauto              # Duplikat-Automatik
.venv/bin/python -m musiksort.tags --zaehlen       # Sterne/Sprachen, die noch in die Dateien müssen
.venv/bin/python -m musiksort.startmodell exportieren  # eigene Bewertungen als Startmodell (anonym)
sh scripts/check.sh                                # Syntax, JavaScript, Tests
```

## Lizenz

Der Code steht unter der **GNU AGPL-3.0** (siehe `LICENSE`), weil er Essentia (AGPL-3.0) einbindet.

Fremde Bestandteile:

- **Essentia-Modelle** (Discogs-EffNet, genre_discogs400) der Music Technology Group, Universitat Pompeu Fabra:
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) – **nur nicht-kommerziell**.
  Deshalb nicht im Repo; wer sie lädt, ist an diese Lizenz gebunden.
- **Startmodell** (`startmodell/`): mit den Essentia-Modellen berechnet, daher ebenfalls
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), nur nicht-kommerziell (siehe `startmodell/LIZENZ.md`).
- **Whisper**-Gewichte (OpenAI) und faster-whisper: MIT
- **Font Awesome Free**: Icons CC BY 4.0, Schrift SIL OFL 1.1, Code MIT (`musiksort/static/fa/LICENSE.txt`)
- **AcoustID**: der Webdienst ist für nicht-kommerzielle Nutzung frei; jede Installation braucht ihren eigenen Schlüssel.
