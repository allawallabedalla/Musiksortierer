"""Tests laufen ohne die lokalen Ordner-Einstellungen der eigenen Sammlung (config.EINSTELLUNGEN)."""
import os
import tempfile

os.environ["MUSIKSORT_EINSTELLUNGEN"] = os.path.join(tempfile.gettempdir(), f"musiksort-test-{os.getpid()}-fehlt.json")
