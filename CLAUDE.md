# Projektregeln

Das Programm liegt in `fruit-ninja-cv/` (Basis: Noahs Projekt). Vorgaben aus `aufgabenstellung.md`, die bei jeder Änderung gelten:

- **Testgetrieben:** Für jede Änderung zuerst Testcases in `fruit-ninja-cv/tests/` ergänzen oder anpassen, dann implementieren. Logik headless testbar halten (keine Kamera/GUI in Tests).
- **Alle Tests ausführen:** Nach jeder Änderung `python -m pytest` in `fruit-ninja-cv/`. Eine Aufgabe ist erst fertig, wenn alle Tests grün sind.
- **Architekturentscheidungen** in `Architektur.md` festhalten.
- **Logging** über `logging_config.py` (`logs/fruit-ninja.log`), wichtige Abläufe loggen, nicht pro Frame.
- **Iterationen:** Arbeit läuft über Issues im Milestone der aktuellen Iteration, ein Branch pro Issue, Pull Request gegen `main`.
- Maßgeblich ist `aufgabenstellung.md` (kleingeschrieben); keine zweite Datei mit gleichem Namen in anderer Schreibweise anlegen (Windows).
- Kamera an diesem Windows-Laptop: ELP-USB-Kamera an Index 2; mit MSMF-Backend in 1080p MJPG 30 fps, 720p liefert nur 9 fps.
