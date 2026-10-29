#!/bin/sh
# Vor jedem Commit/Push: Syntax, JavaScript der Seiten, Tests. Exit-Code zählt (nicht per Pipe verschlucken).
set -e
cd "$(dirname "$0")/.."
.venv/bin/python -m compileall -q musiksort tests scripts
for f in musiksort/static/*.html; do
  awk '/<script>/{a=1;next}/<\/script>/{a=0}a' "$f" > "$HOME/.cache/musiksortierer/check.js"
  node --check "$HOME/.cache/musiksortierer/check.js"
done
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp "$HOME/.cache/musiksortierer/pytest" tests
