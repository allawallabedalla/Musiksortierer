# Startmodell – Lizenz und Inhalt

Die Dateien in diesem Ordner (`steckbriefe.npz`, `modell.npz`, `info.json`) stehen unter
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.de):
nur nicht-kommerziell, Namensnennung, Weitergabe unter gleichen Bedingungen.

Grund: Die Klang-Steckbriefe sind Ausgaben des Discogs-EffNet-Modells (Essentia, Music Technology Group,
Universitat Pompeu Fabra), das selbst unter CC BY-NC-SA 4.0 steht.

Namensnennung: Musiksortierer-Startmodell, Bewertungen von allawallabedalla; Klang-Steckbriefe mit
Discogs-EffNet (Alonso-Jiménez et al., Essentia, MTG-UPF).

Inhalt je Beispiel: 1280 Klang-Werte (normiert), Länge in Sekunden, Gesangssprache (8 Werte) und Genre.
Keine Dateinamen, Interpreten, Titel, Lied-IDs oder Zeitpunkte; Kopien desselben Lieds zählen einmal,
die Reihenfolge ist zufällig. Wer dieselbe Aufnahme besitzt, könnte über den Klang-Steckbrief erkennen,
dass sie dabei ist, und welches Genre sie bekam.

Erzeugt mit `python -m musiksort.startmodell exportieren`, geladen mit `python -m musiksort.startmodell laden`.
