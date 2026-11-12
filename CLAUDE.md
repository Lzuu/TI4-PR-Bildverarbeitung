# Projektregeln

Das Programm liegt in `fruit-ninja-cv/` (Basis: Noahs Projekt). Vorgaben aus `aufgabenstellung.md`, die bei jeder Änderung gelten:

- **Testgetrieben:** Für jede Änderung zuerst Testcases in `fruit-ninja-cv/tests/` ergänzen oder anpassen, dann implementieren. Logik headless testbar halten (keine Kamera/GUI in Tests).
- **Alle Tests ausführen:** Nach jeder Änderung `python -m pytest` in `fruit-ninja-cv/`. Eine Aufgabe ist erst fertig, wenn alle Tests grün sind.
- **Architekturentscheidungen** in `Architektur.md` festhalten.
- **Logging** über `logging_config.py` (`logs/fruit-ninja.log`), wichtige Abläufe loggen, nicht pro Frame.
- **Iterationen:** Arbeit läuft über Issues im Milestone der aktuellen Iteration, ein Branch pro Issue, Pull Request gegen `main`.
- Maßgeblich ist `aufgabenstellung.md` (kleingeschrieben); keine zweite Datei mit gleichem Namen in anderer Schreibweise anlegen (Windows).
- Kamera an diesem Windows-Laptop: ELP-USB-Kamera an Index 2; mit MSMF-Backend in 1080p MJPG 30 fps, 720p liefert nur 9 fps.

# Schnell arbeiten

Diese Übersicht ersetzt das Erkunden des Repos. Erst hier nachsehen, dann gezielt die genannte Datei öffnen.

## Befehle

Alle Befehle laufen in `fruit-ninja-cv/`.

- Tests einer Datei, während der Arbeit: `python -m pytest tests/test_game.py`
- Ein einzelner Test: `python -m pytest tests/test_game.py -k combo`
- Alle Tests, vor dem Abschluss: `python -m pytest` (345 Tests, rund 5 s)
- Hat `python` kein OpenCV (`ModuleNotFoundError: cv2`), die lokale Umgebung nehmen: `.venv\Scripts\python -m pytest` (macOS/Linux: `.venv/bin/python`). Fehlt sie: `python -m venv .venv`, dann `pip install -r requirements-dev.txt`.
- GitHub: `gh issue view <nr>`, `gh pr create`. Das Spiel selbst (`python main.py`) braucht Kamera und Fenster und wird von Agenten nicht gestartet.

## Was nicht gelesen werden muss

- `Architektur.md` hat rund 570 Zeilen und ist überwiegend Verlauf. Nicht ganz lesen. Per Suche den Abschnitt zum Thema öffnen; neue Entscheidungen als weiteren Punkt unter „Architekturentscheidungen (ADR-Kurzform)“ anhängen.
- `docs/`, `suit_assets/*.png`, `tests/data/*.jpg`, `tools/doomscroll/` sind für die meisten Aufgaben ohne Belang.
- `models/`, `assets/`, `faces/`, `logs/`, `settings.json`, `highscores.json` entstehen zur Laufzeit und stehen nicht in Git.

## Module und zugehörige Tests

| Modul | Aufgabe | Tests |
|---|---|---|
| `main.py` (990 Zeilen) | Einstieg, Layout- und Zeichenfunktionen, `_run_game_loop` mit allen Zuständen | `test_main_loop.py`, `test_menu.py`, `test_screen.py`, `test_perf_loop.py` |
| `config.py` | alle Konstanten | keine eigenen |
| `game.py`, `fruit.py`, `blade.py` | Spiellogik, Wurfphysik, Klinge | `test_game.py`, `test_fruit.py`, `test_blade.py` |
| `difficulty.py`, `colors.py`, `dwell.py`, `keyboard.py`, `highscore.py`, `heart.py` | Presets, Stiftfarben, Dwell-Auswahl, Bildschirmtastatur, Bestenliste, Herz-Sprite | gleichnamige `test_*.py` |
| `utils.py` | `point_segment_distance`, `draw_text` | `test_geometry.py` |
| `marker_tracker.py` | Stift per HSV finden, Spitze bestimmen | `test_marker_tracker.py` |
| `background.py` | Spielermaske (Greenscreen, Referenzbild), Weltraum-Compositing | `test_background.py` |
| `yolo_seg.py` | Personenmaske per YOLO, lädt erst bei Bedarf | `test_yolo_seg.py` |
| `pose_tracking.py` | BlazePose-Gelenke, optischer Fluss | `test_pose_tracking.py`, `test_pose_models.py` |
| `suit.py`, `suit_rig.py` | Raumanzug: Stoff, Sprites, Helm | `test_suit.py`, `test_suit_rig.py` |
| `lightsaber.py` | Lichtschwert am Stift, Griff am Handschuh | `test_lightsaber.py` |
| `face_id.py`, `enroll.py` | Spieler per Gesicht erkennen und einlernen | `test_face_id.py` |
| `cameras.py` | Kameras suchen, öffnen, wechseln; `settings.json` | `test_cameras.py` |
| `perf.py` | Stufenzeiten pro Frame (Taste `p`) | `test_perf.py` |
| `sound.py`, `logging_config.py` | Töne (nur macOS), Logging | keine |

## Ablauf eines Frames in `_run_game_loop`

Kamera lesen → zuschneiden, skalieren, spiegeln → Stift suchen (`tracker.track`) → Gesicht (nur bei Bedarf) → Spielermaske je nach Hintergrundmodus → Compositing → Pose → Anzug → Zustand (`STATE_FACE_SCAN`, `STATE_NAME_ENTRY`, `STATE_MENU`, `STATE_DIFFICULTY`, `STATE_PLAY`, `STATE_GAME_OVER`) → Lichtschwert → Anzeige → Tasten.

Ein Zustand steht jeweils in einem Block `elif state == STATE_...`; nach dem Zustandsnamen suchen statt die Funktion ganz zu lesen. Layout (`*_target`, `*_button`) und Zeichnen (`draw_*`) stehen als eigene Funktionen darüber.

## Tests schreiben

- Test durch die echte Schleife: `tests/loop_harness.py` mit `run_loop(monkeypatch, tmp_path, FakeCamera([...]), frames, BG_MODE=..., ...)`, `green_frame()` und `with_pen(frame, x)`. Beispiele in `test_main_loop.py`.
- `config.WIDTH`, `HEIGHT` und `SCALE` werden erst in `main()` an den Bildschirm angepasst. In Tests gelten 960×540 und `SCALE = 1.0`; ändern nur mit `monkeypatch.setattr(config, ...)`.
- Zeit und Zufall werden hineingereicht, nicht gemessen: `time_fn`/`clock` bei `DwellTracker`, `FaceScanner`, `BodySegmenter`, `FrameStats`; `rng` bei `Game` und `Fruit`; `dt` bei `Game.update`.
- Netze werden durch Attrappen ersetzt (`FakeNet` in `test_yolo_seg.py`, `sample_pose()` in `test_pose_tracking.py`). Nur `test_face_id.py` lädt echte Modelle, einmal pro Datei.
- Keinen Test anlegen, der dasselbe prüft wie ein vorhandener, und keinen, der nur eine Konstante festschreibt.

## Fallstricke

- Zeichenfunktionen ändern das Bild an Ort und Stelle. `clean` ist das unveränderte Kamerabild; Erkennung läuft immer auf `clean`, nie auf `frame`.
- Größen in Pixeln mit `config.SCALE` multiplizieren (`_s()` in `main.py`).
- YOLO- und Anzug-Zwischenstände gehören zu einer Quelle `(Kameraindex, mode_session)`; eine neue Quelle verwirft sie.
- Ein Fehler in einem optionalen Feature (YOLO, Pose, Gesicht, Ton) schaltet das Feature ab und wird einmal geloggt; das Spiel läuft weiter.
- Offene Review-Punkte stehen in den Issues #54 (Performance) und #55 (Fehler, Aufräumen).
