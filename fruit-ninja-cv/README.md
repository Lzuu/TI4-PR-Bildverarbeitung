# Fruit Ninja CV

Ein Fruit-Ninja-Klon, gesteuert mit einem **pinken Stift als Schwert** vor einer
Webcam — reines **OpenCV + Python**. Früchte werden von unten ins Bild geworfen und
fliegen im Bogen auf; du zerschneidest sie, indem du den Stift über sie führst
(**Berührung reicht**). Bomben darfst du **nicht** treffen.

Projekt im Modul **Bildverarbeitung** (TI, Semester 4 – HS Albstadt-Sigmaringen).
Entwickelt **inkrementell-iterativ** und **test-case-basiert** (siehe unten).

## Aufgabenstellung
Entwicklung einer interaktiven Anwendung, die mit einer **Webcam** und
**OpenCV (Python)** ein Live-Kamerabild auswertet und daraus eine Steuerung ableitet.
Umgesetzt als Spiel: Erkennung und Verfolgung eines farbigen Markers (pinker Stift),
Ableiten von Bewegungen, Spiel-Physik und Rückmeldung in Echtzeit.

### Anforderungen
- Live-Bild der Webcam erfassen und anzeigen.
- Den pinken Stift farbbasiert erkennen und seine Position verfolgen.
- Berührung des Stifts mit einer Frucht erkennt den Schnitt.
- Früchte mit Wurf-Physik von unten einwerfen, im Bogen fliegen lassen, zerschneiden.
- Punkte-, Leben- und Game-Over-Logik.
- Echtzeitfähig; test-case-basiert entwickelt.

## Bildverarbeitungs-Pipeline (das Lehrreiche daran)
1. **Capture & Preprocessing** — Frame holen, auf feste Größe skalieren, spiegeln.
2. **Farb-Segmentierung** — BGR → HSV, `cv2.inRange` mit dem Pink-Bereich des
   Stifts, Morphologie (Open/Close) gegen Rauschen.
3. **Tracking** — größte Kontur (`findContours`), Schwerpunkt via `cv2.moments`.
4. **Physik** — Früchte als Projektile (Wurf von unten, Schwerkraft, Parabel).
5. **Kollision** — Abstand Frucht-Mittelpunkt ↔ Klingen-Segment (Linie-Kreis).
6. **Rendering** — prozedurale Früchte, Klingen-Trail, HUD-Overlay.

## Installation
```bash
cd fruit-ninja-cv
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Starten
```bash
python main.py
```

Kamera: Standard ist `CAMERA_INDEX = 1` (MacBook FaceTime HD). Für eine andere
Kamera (z. B. iPhone/Continuity = 0, USB-Cam) den Index in `config.py` ändern.

## Ablauf & Steuerung
1. **Start/Kalibrierung:** Am besten **direkt auf den pinken Stift klicken** — damit
   wird exakt seine Farbe gelernt und das Spiel startet. Alternativ **ENTER**
   (Standard-Pink) oder Stift in die Box + **SPACE**. Oben rechts zeigt die
   **Masken-Vorschau**, ob der Stift sauber erkannt wird (nur er sollte weiß sein).
2. **Spielen:** Stift über die Früchte führen (**Berührung reicht**) → Punkte.
   Verpasste Früchte kosten ein Leben. Bomben berühren = sofort vorbei.

| Taste / Aktion | Funktion |
|----------------|----------|
| **Klick auf Stift** | Kalibrieren & starten (präziseste Methode) |
| `SPACE` | Kalibrieren über die Box & starten |
| `ENTER` | Direkt starten mit Standard-Pink |
| `r` | Neustart (im Game-Over-Screen) |
| `c` | Stift neu kalibrieren (jederzeit) |
| `d` | Masken-Debugfenster an/aus |
| `q` | Beenden |

## Tests (test-case-basiert)
Die Spiel-Logik ist durch eine **pytest**-Suite abgedeckt (Geometrie, Marker-Tracking,
Frucht-Physik, Klinge, Spiel-Logik). Die Tests laufen headless (ohne Kamera/GUI).

```bash
pip install -r requirements-dev.txt
pytest
```

| Testdatei | Deckt ab |
|-----------|----------|
| `tests/test_geometry.py` | Linie-Kreis-Abstand (Kollision) |
| `tests/test_marker_tracker.py` | Pink-Erkennung, Kalibrierung, Hintergrund ignorieren |
| `tests/test_fruit.py` | Wurf-Physik: Start unten, Bogen im Bild, Miss nur beim Fallen |
| `tests/test_blade.py` | Trail, Geschwindigkeit, aktueller Punkt |
| `tests/test_game.py` | Berührungs-Schnitt, Bombe, Miss, Spawning, Reset |

## Entwicklung (inkrementell-iterativ über GitHub-Issues)
Die Umsetzung erfolgt in **Inkrementen**, die als **GitHub-Issues** abgebildet werden.
Jedes Issue = ein funktionaler Zuwachs, der implementiert, getestet und abgeschlossen
wird. Siehe den [Issues-Tab](../../issues) des Repositories.

## Tuning
Alle Stellschrauben stehen in `config.py`:
- `CAMERA_INDEX` — welche Kamera (1 = MacBook, 0 = iPhone/Continuity).
- `MARKER_HSV_LOWER` / `MARKER_HSV_UPPER` — Standard-Pink-Bereich.
- `H_TOLERANCE` / `S_FLOOR` / `V_FLOOR` — Farbton-Fenster & Mindest-Sättigung/-Helligkeit.
- `MIN_MARKER_AREA` — kleinere Werte erkennen auch dünne/kleine Stifte.
- `FRUIT_LAUNCH_VY` — wie hoch/kräftig die Würfe fliegen (Betrag größer = höher).
- `FRUIT_DRIFT_VX` — wie schräg/seitlich geworfen wird.
- `GRAVITY` — Tempo der Flugbahn.
- `SPAWN_INTERVAL` / `BURST_WEIGHTS` — Wurf-Frequenz & Mehrfachwürfe.
- `BOMB_PROBABILITY` — Anteil Bomben (0 = keine Bomben).

## Dateien
| Datei | Inhalt |
|-------|--------|
| `main.py` | Game-Loop, States (Start/Play/GameOver), Tasten, Klick-Kalibrierung |
| `config.py` | Alle Konstanten |
| `marker_tracker.py` | Pink-Farb-Segmentierung + Kalibrierung + Tracking |
| `blade.py` | Klingen-Trail + Geschwindigkeit |
| `fruit.py` | Frucht-/Bomben-Projektil-Physik + Zeichnung |
| `game.py` | Spawning, Kollision, Score, Leben |
| `utils.py` | Geometrie (Linie-Kreis) + Text-HUD |
| `tests/` | pytest-Suite |
