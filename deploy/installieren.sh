#!/bin/sh
# User-Units des Musiksortierers installieren (kopiert, ersetzt vorhandene) und Timer einschalten.
set -e
ziel="$HOME/.config/systemd/user"
mkdir -p "$ziel" "$HOME/.config/musiksortierer" "$HOME/.local/share/musiksortierer" "$HOME/.cache/musiksortierer"
cp "$(dirname "$0")"/musiksort-*.service "$(dirname "$0")"/musiksort-*.timer "$(dirname "$0")"/musiksort.slice "$ziel/"
systemctl --user daemon-reload
systemctl --user enable musiksort-web.service musiksort-bewertungen.timer musiksort-analyse.timer musiksort-cluster.timer musiksort-suchen.timer musiksort-sicherung.timer musiksort-sprache.timer
