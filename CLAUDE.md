# Projektregeln

Vorgaben aus `aufgabenstellung.md`, die bei jeder Änderung gelten:

- **Testgetrieben:** Für jede Änderung zuerst Testcases in `tests/` ergänzen oder anpassen, dann implementieren. Bildverarbeitung mit synthetischen Bildern testen (Hilfen in `tests/conftest.py`), Hardware-Tests mit `@pytest.mark.camera` markieren.
- **Alle Tests ausführen:** Nach jeder Änderung `python -m pytest`. Eine Aufgabe ist erst fertig, wenn alle Tests grün sind.
- **Architekturentscheidungen** als neuen AE-Eintrag in `Architektur.md` festhalten.
- **Logging:** Über `logging.getLogger(<modul>)`, eingerichtet mit `log_setup.setup_logging`. Wichtige Abläufe loggen, nie pro Frame.
- **Iterationen:** Arbeit läuft über Issues im Milestone der aktuellen Iteration, ein Branch pro Issue, Pull Request gegen `main`.
- Testbare Logik als Funktionen ohne Kamera/Fenster schreiben, `main()` nur verdrahten lassen.
- Kamera: ELP an Index 2, MSMF-Backend, in 1080p aufnehmen (720p liefert nur 9 fps).
