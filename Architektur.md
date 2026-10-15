# Architektur

Dieses Dokument beschreibt den Aufbau des Programms und hält alle Architekturentscheidungen fest.
**Jede neue Architekturentscheidung wird unten als neuer Eintrag (AE-Nummer) ergänzt** – auch wenn sie eine frühere ersetzt. Ersetzte Einträge bleiben stehen und werden als „ersetzt durch AE-x“ markiert.

## Überblick

| Datei | Aufgabe |
|---|---|
| `camera.py` | Bilderfassung: Kamera öffnen und konfigurieren (Teilaufgabe 1), eigenständig startbar als Kameratest |
| `fruit_ninja.py` | Stifterkennung, Tracking, Gestenerkennung und Spiel (Teilaufgaben 2–5) |
| `log_setup.py` | Einrichtung der Logdateien |
| `tests/` | Testcases (pytest), eine Datei pro Teilaufgabe |

## Ablauf pro Kamerabild

```
Kamera (1920x1080, MJPG, 30 fps)
  → spiegeln, auf 1280x720 verkleinern                  (Spielkoordinaten = Kamerakoordinaten)
  → to_hsv: 640x360, HSV, Maske farbiger Pixel
  → find_pen: Rückprojektion des Farbmodells → Maske → größte Fläche → Schwerpunkt
  → update_blade: Spur fortführen, Sprünge verwerfen, Geschwindigkeit prüfen → Klinge (Strecke) oder None
  → Spiellogik: segment_hits(Klinge, Frucht) → Punkte / Bombensperre
  → Zeichnen: Früchte, Effekte, Klingenspur, Texte, optional Maske
```

## Zustände des Spiels

```
calibrate ──Leertaste, Kalibrierung ok──▶ verify ──Leertaste──▶ menu ──Melone/Leertaste──▶ countdown ──3 s──▶ play
    ▲                                       │ r                  │ k                                            │ 30 s
    └───────────────────────────────────────┴────────────────────┘                                    menu ◀──┘
```

## Architekturentscheidungen

### AE-1: Kamera über Media Foundation (MSMF) in 1080p
- **Kontext:** Unter DirectShow ignoriert die ELP-USBFHD01 die MJPG-Einstellung und liefert bei 1080p nur 5 fps, bei 720p (auch unter MSMF) nur 9 fps.
- **Entscheidung:** Backend `CAP_MSMF`, Aufnahme in 1920x1080 MJPG mit 30 fps, Verkleinern auf 1280x720 im Programm.
- **Folge:** Volle 30 fps. Das Verkleinern kostet etwas Rechenzeit, die aber gering ist.

### AE-2: Stifterkennung über Farbe mit Kalibrierung beim Start
- **Kontext:** Nur der Stift darf schneiden, nicht jede Bewegung. Feste Farbschwellen (z. B. nur Orange) sind lichtabhängig und an einen bestimmten Stift gebunden.
- **Entscheidung:** Beim Start wird der Gegenstand in einem Feld kalibriert. Aus den Farben im Feld und im restlichen Bild entsteht ein H-S-Histogramm mit der Wahrscheinlichkeit „Farbe gehört zum Gegenstand“. Gesucht wird per Histogramm-Rückprojektion; die größte zusammenhängende Fläche ist der Stift.
- **Begründung:** Funktioniert mit jedem farbigen Gegenstand. Farbton und Sättigung hängen wenig von der Helligkeit ab. Farben, die im Hintergrund häufig sind, werden automatisch abgewertet. Die Kalibrierung dauert nur einen Tastendruck.
- **Folge:** Graue, schwarze oder weiße Gegenstände sind nicht nutzbar. Ragt beim Kalibrieren die Hand ins Feld, wird Hautfarbe mitgelernt.

### AE-3: Farbsuche auf halber Auflösung
- **Entscheidung:** Erkennung auf 640x360.
- **Begründung:** Gemessen etwa 3–4 ms pro Bild, die Genauigkeit reicht für das Spiel (Positionsfehler unter 5 px, siehe `tests/test_detection.py`).

### AE-4: Schnitt = Strecke zwischen zwei Frames mit Mindestgeschwindigkeit
- **Entscheidung:** Die Klinge ist die Strecke von der vorherigen zur aktuellen Stiftposition. Sie zählt nur, wenn der Stift mindestens `MIN_SLICE_SPEED` px/s schnell war. Getroffen ist eine Frucht, wenn die Strecke ihren Kreis schneidet. Sprünge über `MAX_JUMP` px gelten als Fehlerkennung.
- **Begründung:** Erfüllt „langsame oder ruhende Bewegungen sind kein Schnitt“. Schnelle Bewegungen, bei denen der Stift zwischen zwei Frames über die Frucht springt, treffen trotzdem.

### AE-5: Testgetriebene Entwicklung mit pytest und GitHub Actions
- **Entscheidung:** Tests liegen in `tests/` (eine Datei pro Teilaufgabe). Bildverarbeitung wird mit synthetischen Bildern getestet, damit die Tests ohne Kamera laufen. Tests, die echte Hardware brauchen, sind mit `@pytest.mark.camera` markiert und laufen nur lokal. Jeder Push und Pull Request startet die Tests auf GitHub (`.github/workflows/tests.yml`); `main` darf nur über Pull Requests mit grünen Tests geändert werden.
- **Folge:** Logik, die getestet werden soll, muss als Funktion ohne Kamera und Fenster aufrufbar sein (z. B. `update_blade`, `plan_wave`).

### AE-6: Logging in eine Datei pro Programmstart
- **Entscheidung:** Python-`logging`. Jeder Start schreibt `logs/<programm>_<datum>_<uhrzeit>.log` (DEBUG), die Konsole zeigt INFO. Es werden die letzten 20 Logs je Programm behalten. Geloggt werden Kameraeinstellungen, Kalibrierergebnis, Zustandswechsel, Treffer, Rundenergebnis und alle 5 s Bildrate, Verarbeitungszeit und Erkennungsrate.
- **Folge:** Keine Logausgabe pro Frame, damit die Dateien lesbar bleiben und das Spiel nicht ausgebremst wird.

### AE-7: Iterationen über GitHub Milestones
- **Entscheidung:** Jede Iteration ist ein Milestone. Aufgaben sind Issues im Milestone, jede Aufgabe wird auf einem eigenen Branch bearbeitet und per Pull Request (mit Tests) in `main` gebracht.

## Offene Entscheidungen

- **Netzwerk-Mehrspieler (Teilaufgabe 6):** Noch zu entscheiden. Vorschlag: Jeder Spieler hat eigenen Rechner und eigene Kamera, ein Rechner ist Host. Verbindung per TCP-Socket (Python-Standardbibliothek) mit JSON-Nachrichten. Beide Seiten nutzen denselben Zufalls-Seed, damit dieselben Früchte fliegen; ausgetauscht werden nur Start, Punktestand und Ergebnis.
- **Aufteilung von `fruit_ninja.py`:** Sobald Tracking und Netzwerk dazukommen, sollte die Datei in Module aufgeteilt werden (z. B. `detection.py`, `tracking.py`, `game.py`, `network.py`).
