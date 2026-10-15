# Fruit Ninja mit Stift-Tracking

Fruit Ninja, gesteuert mit einem farbigen Stift vor einer USB-Kamera. Aufgabenstellung: [aufgabenstellung.md](aufgabenstellung.md), Aufbau und Entscheidungen: [Architektur.md](Architektur.md).

## Einrichtung

```
python -m pip install -r requirements-dev.txt
```

## Starten

```
python fruit_ninja.py          # Spiel (Kamera-Index 2 = ELP-USB-Kamera)
python fruit_ninja.py 0        # mit anderer Kamera
python camera.py               # nur Kamerabild anzeigen
```

Beim Start wird der Stift kalibriert: Stift ins Feld halten, Leertaste drücken, prüfen, Leertaste. Tasten im Spiel: `m` Maske ein/aus, `k` neu kalibrieren, `q` beenden.

## Tests

```
python -m pytest               # alle Tests (Kameratests werden übersprungen, wenn keine Kamera da ist)
python -m pytest -m "not camera"   # wie auf GitHub
```

## Logs

Jeder Programmstart schreibt eine Logdatei nach `logs/` (z. B. `logs/fruit_ninja_20261015_140312.log`). Bei Fehlern den passenden Ausschnitt ins Issue kopieren.

## Arbeitsweise

Die Vorgaben aus der Aufgabenstellung setzen wir so um:

1. **Iterationen:** Jede Iteration ist ein [Milestone](../../milestones) auf GitHub. Aufgaben sind Issues im Milestone.
2. **Branch pro Aufgabe:** `git switch -c <issue-nummer>-<kurzname>`, z. B. `12-glaettung`.
3. **Test zuerst:** Zuerst einen Testcase in `tests/` schreiben, der die neue Anforderung prüft (und ohne die Änderung fehlschlägt), dann implementieren. Jede Änderung bekommt neue oder angepasste Tests.
4. **Alle Tests ausführen:** Vor jedem Pull Request `python -m pytest` – mit angeschlossener Kamera, damit auch die Kameratests laufen.
5. **Pull Request:** Gegen `main`, Issue mit `Closes #<nr>` verlinken, Checkliste ausfüllen. GitHub führt alle Tests automatisch aus; gemergt wird nur, wenn alle grün sind.
6. **Architekturentscheidungen** kommen als neuer Eintrag in [Architektur.md](Architektur.md).
7. **Logging:** Neue Abläufe schreiben sinnvolle Log-Meldungen (`log.info` / `log.debug`), aber nicht pro Frame.
