# Fruit Ninja CV

Ein Fruit-Ninja-Klon, gesteuert mit einem **farbigen Stift als Schwert** vor einer
Webcam — reines **OpenCV + Python**. Früchte werden von unten ins Bild geworfen und
fliegen im Bogen auf; du zerschneidest sie, indem du den Stift über sie führst
(**Berührung reicht**). Bomben darfst du **nicht** treffen.

Projekt im Modul **Bildverarbeitung** (TI, Semester 4 – HS Albstadt-Sigmaringen).
Entwickelt **inkrementell-iterativ** und **test-case-basiert**.

## Features
- 🎯 **Farb-Tracking** mit festen Bereichen (keine Kalibrierung nötig).
- 🏠 **Startmenü:** standardmäßig **Pink** aktiv (nur Pink wird getrackt). Spiel per
  Stift (**Dwell**) starten; Quit ebenfalls per Stift.
- 🖥️ **Fullscreen & responsive:** startet im Vollbild; Layout **und Physik**
  skalieren mit der Bildschirmauflösung (kein grauer Rand).
- 🖱️ **Stiftfarbe als Einstellung:** hinter einem **„Stiftfarbe"-Button** im Menü —
  erst per **Maus-Klick** öffnen, dann Farbe (Pink/Gelb/Grün) wählen. Nur nötig,
  wenn man die Farbe wirklich ändern will.
- 🕹️ **VR-Auswahl per Dwell:** Stift 3 s in einen Kreis halten → Ladering füllt
  sich → bestätigt (Start, Schwierigkeit, Game-Over-Felder – ohne Tastatur).
- ❤️ **Herz-Icon (Sprite)** als Lebensanzeige – gerenderte Herzkurve mit Verlauf
  und Glanz.
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

## Spieler per Gesicht erkennen
Der **erste Bildschirm** ist ein Kreis: Gesicht hineinhalten.
- **Bekanntes Gesicht** → „Hallo, Name!“ → Startmenü.
- **Neues Gesicht** → ein Ladering füllt sich, während ca. 2 s lang Bilder
  aufgenommen werden → Name per **Bildschirmtastatur mit dem Stift** eingeben
  (0,8 s auf eine Taste halten; `OK` bestätigt, leer = „Spieler 1“, „Spieler 2“ …;
  die echte Tastatur geht auch) → Startmenü.
- **„Als Gast“** (Stift 3 s in den Kreis) überspringt den Scan.

Der Name steht im Highscore. Im Startmenü führt **„Spieler wechseln“** zurück zum
Scan. Die Bilder landen in `faces/<Name>/` und bleiben **nur lokal** (nicht in git,
biometrische Daten). Alternativ per Kommandozeile: `python enroll.py "Noah Braun"`.
Die Modelle (YuNet + SFace, ca. 39 MB) lädt das Spiel beim ersten Start nach
`models/`.

## Kamera wählen
Hat der Rechner mehrere Kameras, erscheint im Startmenü und auf dem Gesichtsscan
unten rechts ein **„Kamera“-Button** (auf dem Scan wichtig, falls das Spiel mit der
falschen Kamera startet). Per Maus-Klick öffnet sich eine Auswahl mit Vorschaubildern;
ein Klick auf ein Bild wechselt die Kamera. Die Wahl wird in `settings.json`
gespeichert und beim nächsten Start wieder verwendet. Die Kameras werden einmal
beim Start gesucht (ca. 1 s je Kamera).

## Raumanzug
Ist der Weltraum-Hintergrund aktiv (Taste `b`), trägt der freigestellte Spieler einen
**Raumanzug, der sich mit Armen und Körper bewegt**: Die eigene Silhouette wird zu
weißem, deckendem Anzugstoff (die Falten der echten Kleidung bleiben leicht sichtbar).
**Eigener Anzug:** ein Bild als `textures/suit.png` ablegen – es wird statt des weißen
Stoffs auf den Körper gestreckt (Details in `textures/README.md`). Kopf, Hände und
der Stift bleiben frei. Kragen, Brustpanel und Abzeichen sitzen am Gesicht. Die Details
liegen in `assets/suit_details.png` (wird beim ersten Start erzeugt, darf durch eine
eigene PNG mit Transparenz ersetzt werden). Abschalten: `SUIT_ENABLED = False` in
`config.py`.

## Installation & Start
```bash
cd fruit-ninja-cv
pip install -r requirements.txt
python main.py
```
Kamera: Standard `CAMERA_INDEX = 1` (MacBook FaceTime HD). Andere Kamera in
`config.py` einstellen (z. B. 0 = iPhone/Continuity).

## Ablauf & Steuerung
1. **Startmenü (Vollbild):** Pink ist standardmäßig aktiv. **Start** auswählen,
   indem du den Stift **3 Sekunden** in den Start-Kreis hältst (Ladering bestätigt).
   Andere Stiftfarbe nur bei Bedarf: unten links auf **„Stiftfarbe"** klicken
   (Maus) → Farb-Kachel wählen.
2. **Schwierigkeit wählen:** Stift **3 Sekunden** in den gewünschten Kreis
   (Einfach/Mittel/Schwer) halten.
3. **Spielen:** Stift über die Früchte führen (**Berührung reicht**). Mehrere
   Früchte in einem Swipe geben einen **Kombo-Bonus**. Verpasste Früchte kosten
   ein Leben, Bomben = sofort vorbei.
4. **Game Over:** **Neustart**, **Startmenü** oder **Quit** per Stift-Dwell wählen.

> Alle In-Game-Auswahlen laufen über den Stift (Dwell). Nur die **Stiftfarbe** im
> Startmenü wird per **Maus** gesetzt. Tastatur-Fallback: `q` Beenden, `d` Maske.

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
| `tests/test_cameras.py` | Kamerasuche, Auswahl/Fallback, Wechsel, Speichern, Menü-Layout (ohne echte Kamera) |
| `tests/test_highscore.py` | Top-3 je Schwierigkeit, Ranking, Spielername, alte Dateien, defekte Datei |
| `tests/test_face_id.py` | Gesichtsdetektion, Wiedererkennung auf anderem Foto, Gast, Scan im Kreis, Einlernen, Startablauf |
| `tests/test_suit.py` | Spieler-Box/Schwerpunkt, Anzug folgt dem erhobenen Arm, Kopf/Hände/Stift bleiben frei, Stoff mit Falten und Rundung, Details am Gesicht, Overlay, Glättung, Sprite |
| `tests/test_keyboard.py` | Bildschirmtastatur: Layout, Großschreibung, Löschen, Leerzeichen, Länge, echte Tastatur |

## Tuning
Marker/Kamera & allgemeine Konstanten in `config.py`:
- `CAMERA_INDEX` — Standard-Kamera, solange keine im Menü gewählt wurde.
- `MIN_MARKER_AREA` / `MORPH_KERNEL` — Erkennung dünner Stifte / Rauschfilter.
- `DWELL_SECONDS` / `SELECT_RADIUS` — Dauer & Größe der Dwell-Auswahl.
- `COMBO_WINDOW_FRAMES` / `COMBO_MIN` / `COMBO_BONUS_PER_FRUIT` — Kombo-Regeln.

Stiftfarben in `colors.py`; schwierigkeitsabhängiges **Tempo** in `difficulty.py`.

## Dateien
| Datei | Inhalt |
|-------|--------|
| `main.py` | Game-Loop, States (FaceScan/NameEntry/Menu/Difficulty/Play/GameOver), Dwell-Auswahl |
| `config.py` | Gemeinsame Konstanten (Kamera, Dwell, Kombos, Highscore) |
| `colors.py` | Feste Stiftfarben Pink/Gelb/Grün (HSV + UI-Farbe) |
| `difficulty.py` | Schwierigkeitsgrade (Tempo; immer 3 Leben) |
| `dwell.py` | VR-Dwell-Auswahl (3 s Halten → bestätigt) |
| `marker_tracker.py` | Farb-Segmentierung (eine/mehrere Farben), Tracking |
| `blade.py` | Klingen-Trail + Geschwindigkeit |
| `fruit.py` | Frucht-/Bomben-Projektil-Physik + Zeichnung |
| `game.py` | Spawning, Kollision, Score, Leben, Kombos |
| `heart.py` | Herz-Sprite (parametrische Herzkurve) für die Lebensanzeige |
| `background.py` | Spieler freistellen, Weltraum-Hintergrund |
| `suit.py` | Raumanzug aus der Silhouette (bewegt sich mit) |
| `cameras.py` | Kameras finden, öffnen, wechseln, Auswahl speichern |
| `highscore.py` | Top-3 je Schwierigkeit mit Spielername (JSON-Persistenz) |
| `face_id.py` | Spielererkennung (YuNet + SFace), Scan im Kreis, Einlernen |
| `keyboard.py` | Bildschirmtastatur für die Namenseingabe per Stift |
| `enroll.py` | Spieler einlernen (`faces/<Name>/`) |
| `sound.py` | Soundeffekte (synthetisiert, afplay) |
| `logging_config.py` | Logfiles (rotierend, `logs/`) |
| `utils.py` | Geometrie (Linie-Kreis) + Text-HUD |
| `tests/` | pytest-Suite |
