# Fruit Ninja CV

Ein Fruit-Ninja-Klon, gesteuert mit einem **farbigen Stift als Schwert** vor einer
Webcam — reines **OpenCV + Python**. Früchte werden von unten ins Bild geworfen und
fliegen im Bogen auf; du zerschneidest sie, indem du den Stift über sie führst
(**Berührung reicht**). Bomben darfst du **nicht** treffen.

Projekt im Modul **Bildverarbeitung** (TI, Semester 4 – HS Albstadt-Sigmaringen).
Entwickelt **inkrementell-iterativ** und **test-case-basiert**.

## Features
- 🎯 **Farb-Tracking** mit festen Bereichen (keine Kalibrierung nötig).
- 🖍️ **Drei Stiftfarben** wählbar: **Pink**, **Gelb**, **Grün**.
- 🕹️ **VR-Auswahl per Dwell:** Stift 3 s in einen Kreis halten → Ladering füllt
  sich → bestätigt (für Farbe **und** Schwierigkeit, ganz ohne Tastatur).
- 🍉 **Wurf-Physik:** Früchte fliegen von unten im Bogen herein.
- ✂️ **Schneiden durch Berührung** (keine Mindestgeschwindigkeit).
- 🎚️ **Drei Schwierigkeitsgrade** – schnellere Früchte = schwerer (immer **3 Leben**).
- 🏆 **Leaderboard je Schwierigkeit** (Top-3, persistent).
- 🔥 **Kombos** à la Fruit Ninja: mehrere Früchte in einem Swipe → Bonuspunkte.
- 🔊 **Soundeffekte** (synthetisiert), inkl. realistischem Schnitt-Swoosh.
- 🧾 **Logfiles** fürs Debugging, **pytest**-Tests für die Logik.

## Bildverarbeitungs-Pipeline (das Lehrreiche daran)
1. **Capture & Preprocessing** — Frame holen, auf feste Größe skalieren, spiegeln.
2. **Farb-Segmentierung** — BGR → HSV, `cv2.inRange` mit festen Farbbereichen
   (Pink/Gelb/Grün, siehe `colors.py`), Morphologie (Open/Close) gegen Rauschen.
3. **Tracking** — größte Kontur (`findContours`), Schwerpunkt via `cv2.moments`.
4. **Physik** — Früchte als Projektile (Wurf von unten, Schwerkraft, Parabel).
5. **Kollision** — Abstand Frucht-Mittelpunkt ↔ Klingen-Segment (Linie-Kreis).
6. **Rendering** — prozedurale Früchte, Klingen-Trail, HUD, Dwell-Ladekreise.

## Installation & Start
```bash
cd fruit-ninja-cv
pip install -r requirements.txt
python main.py
```
Kamera: Standard `CAMERA_INDEX = 1` (MacBook FaceTime HD). Andere Kamera in
`config.py` einstellen (z. B. 0 = iPhone/Continuity).

## Ablauf & Steuerung
1. **Farbe wählen:** Den Stift **3 Sekunden** in den Kreis deiner Stiftfarbe
   (Pink/Gelb/Grün) halten — der Ladering bestätigt die Auswahl.
2. **Schwierigkeit wählen:** Ebenso **3 Sekunden** in den gewünschten Kreis
   (Einfach/Mittel/Schwer) halten.
3. **Spielen:** Stift über die Früchte führen (**Berührung reicht**). Mehrere
   Früchte in einem Swipe geben einen **Kombo-Bonus**. Verpasste Früchte kosten
   ein Leben, Bomben = sofort vorbei.

| Taste | Funktion |
|-------|----------|
| `r` | Neustart (gleiche Schwierigkeit) |
| `m` | Zurück zur Auswahl (Farbe/Schwierigkeit) |
| `d` | Masken-Debugfenster an/aus |
| `q` | Beenden |

> Die Auswahl läuft komplett über den Stift (Dwell), nicht über die Tastatur.

## Tests (test-case-basiert)
```bash
pip install -r requirements-dev.txt
pytest
```

| Testdatei | Deckt ab |
|-----------|----------|
| `tests/test_geometry.py` | Linie-Kreis-Abstand (Kollision) |
| `tests/test_marker_tracker.py` | Farberkennung Pink/Gelb/Grün + Union |
| `tests/test_colors.py` | Farbdefinitionen |
| `tests/test_dwell.py` | VR-Dwell-Auswahl (Zeitlogik) |
| `tests/test_fruit.py` | Wurf-Physik: Start unten, Bogen im Bild, Miss beim Fallen |
| `tests/test_blade.py` | Trail, Geschwindigkeit, aktueller Punkt |
| `tests/test_game.py` | Berührungs-Schnitt, Bombe, Miss, Spawning, Reset, Kombos |
| `tests/test_difficulty.py` | 3 Leben überall, härter = schneller |
| `tests/test_highscore.py` | Top-3 je Schwierigkeit, Ranking, defekte Datei |

## Tuning
Marker/Kamera & allgemeine Konstanten in `config.py`:
- `CAMERA_INDEX` — welche Kamera.
- `MIN_MARKER_AREA` / `MORPH_KERNEL` — Erkennung dünner Stifte / Rauschfilter.
- `DWELL_SECONDS` / `SELECT_RADIUS` — Dauer & Größe der Dwell-Auswahl.
- `COMBO_WINDOW_FRAMES` / `COMBO_MIN` / `COMBO_BONUS_PER_FRUIT` — Kombo-Regeln.

Stiftfarben in `colors.py`; schwierigkeitsabhängiges **Tempo** in `difficulty.py`.

## Dateien
| Datei | Inhalt |
|-------|--------|
| `main.py` | Game-Loop, States (Color/Difficulty/Play/GameOver), Dwell-Auswahl |
| `config.py` | Gemeinsame Konstanten (Kamera, Dwell, Kombos, Highscore) |
| `colors.py` | Feste Stiftfarben Pink/Gelb/Grün (HSV + UI-Farbe) |
| `difficulty.py` | Schwierigkeitsgrade (Tempo; immer 3 Leben) |
| `dwell.py` | VR-Dwell-Auswahl (3 s Halten → bestätigt) |
| `marker_tracker.py` | Farb-Segmentierung (eine/mehrere Farben), Tracking |
| `blade.py` | Klingen-Trail + Geschwindigkeit |
| `fruit.py` | Frucht-/Bomben-Projektil-Physik + Zeichnung |
| `game.py` | Spawning, Kollision, Score, Leben, Kombos |
| `highscore.py` | Top-3 je Schwierigkeit (JSON-Persistenz) |
| `sound.py` | Soundeffekte (synthetisiert, afplay) |
| `logging_config.py` | Logfiles (rotierend, `logs/`) |
| `utils.py` | Geometrie (Linie-Kreis) + Text-HUD |
| `tests/` | pytest-Suite |
