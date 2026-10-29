"""Ausfall eines Dienstes melden (OnFailure der systemd-Units).

Mit PUSHOVER_TOKEN und PUSHOVER_USER (Umgebung oder ~/.env) als Push aufs Handy, sonst nur ins Journal.

  python -m musiksort.meldung <unit>
"""
import sys
import urllib.parse
import urllib.request

from . import config


def melden(unit):
    titel, text = f"Musiksortierer: {unit} gescheitert", f"journalctl --user -u {unit}"
    token, user = config.env("PUSHOVER_TOKEN"), config.env("PUSHOVER_USER")
    if not (token and user):
        print(f"{titel} ({text}); keine Push-Zugangsdaten, nur ins Journal")
        return False
    daten = urllib.parse.urlencode({"token": token, "user": user, "title": titel, "message": text}).encode()
    with urllib.request.urlopen("https://api.pushover.net/1/messages.json", daten, timeout=30) as r:
        return r.status == 200


if __name__ == "__main__":
    melden(sys.argv[1] if len(sys.argv) > 1 else "unbekannt")
