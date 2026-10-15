# Architektur

Architekturentscheidungen und Aufbau des Fruit-Ninja-CV-Spiels
(`fruit-ninja-cv/`).

## Überblick
Das Spiel ist in **kleine, fokussierte Module** aufgeteilt (eine Verantwortung je
Datei). `main.py` ist der Einstieg und hält die **Game-Loop** sowie die
**State-Machine** (Kalibrierung → Spiel → Game Over). Die Bildverarbeitung, die
Spiel-Physik und das Rendering sind davon getrennt.

```
Kamera ─► main (Loop, States, Input)
             │
             ├─► MarkerTracker   : Bild → HSV-Segmentierung → Stift-Position
             ├─► Blade           : letzte Positionen = Klinge (Trail, Segment)
             ├─► Game            : Spawning, Physik-Update, Kollision, Score/Leben
             │      └─► Fruit     : Projektil-Physik + prozedurale Zeichnung
             ├─► utils           : Geometrie (Linie-Kreis), Text-HUD
             └─► logging_config  : Logfiles (logs/)
```

## Module & Verantwortlichkeiten
| Modul | Verantwortung |
|-------|---------------|
| `main.py` | Game-Loop, State-Machine (FaceScan/NameEntry/Menu/Difficulty/Play/GameOver), Kamera-Capture, Dwell-Auswahl, Maus-Farbeinstellung, Rendering, Logging-Setup, Fehlerbehandlung |
| `config.py` | Gemeinsame Konstanten (Kamera, Marker, Dwell, Kombos, Highscore, HUD) |
| `colors.py` | Feste Stiftfarben Pink/Gelb/Grün (HSV-Bereich + UI-Farbe) |
| `difficulty.py` | Schwierigkeitsgrade (Tempo; immer 3 Leben) als Presets |
| `dwell.py` | VR-Dwell-Auswahl (Stift n Sekunden auf einem Ziel = Bestätigung) |
| `marker_tracker.py` | HSV-Farb-Segmentierung (eine oder mehrere Farben) + Positionsbestimmung |
| `blade.py` | Kurzer Trail der letzten Marker-Punkte; aktuelles Segment & Geschwindigkeit |
| `fruit.py` | Frucht-/Bomben-Entität: Projektil-Physik + prozedurale Darstellung |
| `game.py` | Spielzustand: Spawning (Bursts), Kollision, Punkte, Leben, Kombos |
| `heart.py` | Herz-Sprite (parametrische Herzkurve, Verlauf+Glanz) für die Leben |
| `highscore.py` | Top-3-Highscore **je Schwierigkeit** mit Spielername (JSON-Persistenz) |
| `face_id.py` | Spielererkennung per Gesicht: YuNet-Detektion, SFace-Merkmale, Abgleich mit `faces/`, Scan im Kreis (`FaceScanner`), Einlernen |
| `keyboard.py` | Bildschirmtastatur (Dwell pro Taste) für den Namen neuer Spieler |
| `enroll.py` | Spieler einlernen: speichert ausgerichtete Gesichtsausschnitte nach `faces/<Name>/` |
| `sound.py` | Soundeffekte (synthetisiert, non-blocking via afplay) |
| `utils.py` | Wiederverwendbare Helfer: Linie-Kreis-Abstand, Text mit Schatten |
| `logging_config.py` | Zentrales Logging (rotierendes Logfile + Konsole) |

## Datenfluss (pro Frame)
1. `main` liest Frame, skaliert auf feste Größe, spiegelt (Mirror).
2. `MarkerTracker.track(frame)` → HSV → `inRange` → Morphologie → größte Kontur →
   Schwerpunkt = Stiftposition (oder `None`).
3. `Blade.add(point)` hält die letzten N Punkte (die Klinge).
4. `Game.update(blade)` wirft Früchte ein, bewegt sie (Schwerkraft), prüft
   **Kollision** (Abstand Frucht-Mittelpunkt ↔ Klingen-Segment ≤ Radius).
5. `Game.draw` + `Blade.draw` + HUD zeichnen ins Frame; `main` zeigt es an.

## Architekturentscheidungen (ADR-Kurzform)
- **Klassische Bildverarbeitung statt ML.** Farb-Segmentierung (HSV) + Konturen
  statt MediaPipe o. Ä. → passt zur Kursintention, transparente Pipeline, kein
  Modell-Overhead.
- **Feste Farbbereiche statt Selbst-Kalibrierung.** Pink/Gelb/Grün sind in
  `colors.py` als feste HSV-Bereiche hinterlegt (Hue als Hauptkriterium, S/V mit
  Mindestwert gegen Haut/heller Wand). Kein fehleranfälliges Live-Kalibrieren mehr.
  **Standard ist Pink, und es wird immer nur die EINE aktive Farbe getrackt**
  (keine Union) → keine Fehldetektion durch andere farbige Objekte.
- **Stiftfarbe als versteckte Einstellung (Maus).** Die Farbe wechselt man nur, wenn
  man will: Sie liegt hinter einem **„Stiftfarbe"-Button** im Menü; erst ein
  **Maus-Klick** öffnet die Farb-Kacheln. Das hält das Menü aufgeräumt und umgeht
  das Henne-Ei-Problem (einen andersfarbigen Stift könnte man nicht per Dwell
  wählen, solange er nicht getrackt wird) — Tracking bleibt einfarbig und robust.
- **Fullscreen & auflösungs-responsive.** Beim Start wird die Bildschirmgröße
  crash-frei über `system_profiler` ermittelt (nicht `tkinter` — das bricht mit
  OpenCVs Cocoa-Fenster ab). Gerendert wird auf ein festes **16:10**-Format
  (`config.ASPECT`), das der nutzbaren MacBook-Vollbildfläche entspricht → **kein
  grauer Rand**. Die 16:9-Webcam wird auf 16:10 **zugeschnitten** (nicht gestaucht)
  → keine Verzerrung. Ein globaler Faktor `config.SCALE = HEIGHT / REF_HEIGHT`
  skaliert UI (`utils.draw_text`, Kreise) **und Physik** (Radius, Wurf, Gravitation
  in `fruit.py`); Radius und Geschwindigkeit/Gravitation werden gemeinsam skaliert,
  sodass die Flugbahn proportional bleibt und die Airtime (in Frames) gleich.
- **Herz als Sprite statt Primitive.** Das Herz wird aus der parametrischen
  Herzkurve mit Farbverlauf, Glanz und Kontur in ein gecachtes BGRA-Sprite gerendert
  (`heart.py`) und per Alpha-Blending gezeichnet → sieht wie ein echtes Spiel-Asset
  aus, ohne externe Bilddatei.
- **VR-Dwell-Auswahl.** Start, Schwierigkeit und die Game-Over-Felder (Neustart/
  Startmenü/Quit) werden ausgewählt, indem der Stift einige Sekunden in einem Kreis
  gehalten wird (Ladering). Entkoppelt in `dwell.py` (zeitbasiert, framerate-
  unabhängig, mit injizierbarer Uhr → gut testbar).
- **Farbige Marker statt Hautfarbe.** Gesättigte Pen-Farben liegen weit vom Hautton
  entfernt → keine Verwechslung mit der Hand.
- **Projektil-Physik.** Früchte starten unter dem unteren Rand mit Aufwärts-
  Geschwindigkeit; Schwerkraft erzeugt die Parabel. Launch-Parameter sind so
  gewählt, dass der Scheitel **im Bild** bleibt (nicht oben abgeschnitten).
- **Kollision per Berührung.** Schneiden erfordert keine Mindestgeschwindigkeit;
  es zählt der Abstand zur Klinge (Punkt bei Ruhe, Segment bei Bewegung).
- **Prozedurale Grafik.** Früchte werden gezeichnet (Kreis + Highlight + Stiel) →
  keine Asset-Abhängigkeiten, läuft überall sofort.
- **State-Machine.** Klar getrennte Zustände (Kalibrierung/Play/GameOver) halten die
  Loop übersichtlich.
- **Zentrale Konfiguration.** Alle Stellschrauben in `config.py` → einfaches Tuning,
  keine Magic Numbers im Code.
- **Schwierigkeitsgrade als Presets.** `difficulty.py` bündelt die variierenden
  Werte je Stufe. Die Schwierigkeit steckt bewusst im **Tempo** der Früchte
  (höhere Gravitation + Wurfgeschwindigkeit + schnelleres Spawnen); **Leben bleiben
  immer 3**. `Game`/`Fruit` bekommen die gewählte Stufe injiziert → keine global
  verstreuten Magic Numbers. Jede Stufe hat ein **eigenes Leaderboard**.
- **Kombos über ein Zeitfenster.** Mehrere Treffer innerhalb `COMBO_WINDOW_FRAMES`
  zählen als ein Swipe; ab `COMBO_MIN` Früchten gibt es Bonuspunkte.
- **Spielererkennung per Gesicht mit vortrainierten Netzen (Issue #28).** Vor der
  Schwierigkeitsauswahl erkennt das Spiel den Spieler und ordnet ihm den Highscore
  zu. Klassische Verfahren (Haar-Cascade + LBPH) gibt es in OpenCV 5 nicht mehr
  (Haar nur noch in OpenCV 4, LBPH nur in opencv-contrib). Das Team hat sich deshalb
  für die in OpenCV 5 enthaltenen DNN-Verfahren entschieden: **YuNet** findet
  Gesichter, **SFace** berechnet aus dem ausgerichteten 112x112-Ausschnitt einen
  Merkmalsvektor; gleiche Person ab Kosinus-Ähnlichkeit 0,363. Genauer und
  lichtrobuster als LBPH, aber eine bewusste Ausnahme von „kein ML“ – nur für die
  Spielererkennung, nicht für die Stifterkennung. Die Modelle (OpenCV Model Zoo,
  ca. 39 MB) werden beim ersten Start nach `models/` geladen und per SHA256 geprüft,
  nicht ins Repo eingecheckt. Ohne Modelle (offline) startet das Spiel direkt im
  Menü und jeder spielt als Gast.
- **Gesichtsscan als erster Bildschirm, Einlernen im Spiel.** Statt eines separaten
  Einlern-Programms hält man das Gesicht in einen Kreis: Ein bekannter Spieler gilt
  nach 5 übereinstimmenden Frames als erkannt (nicht nach einem Einzelbild); bei
  einem unbekannten Gesicht werden 8 ausgerichtete Ausschnitte im Abstand von 0,3 s
  aufgenommen (leichte Kopfbewegungen → robustere Referenz), danach wird der Name
  abgefragt. Gescannt wird beim Start und über „Spieler wechseln“, nicht vor jeder
  Runde. Die Namenseingabe läuft per Stift über eine Bildschirmtastatur (Dwell
  0,8 s statt 3 s pro Taste, sonst zu langsam); die echte Tastatur geht zusätzlich.
  Logik (`FaceScanner`, `NameEntry`) ist von der Darstellung getrennt und mit
  injizierbarer Uhr testbar.
- **Referenzbilder sind biometrische Daten.** `faces/` liegt nur lokal (in
  `.gitignore`); die Tests nutzen gemeinfreie NASA-Fotos (`tests/data/QUELLEN.md`).
- **Highscore als JSON.** `highscore.py` kapselt Laden/Schreiben der Top-3 und ist
  so unabhängig testbar (defekte/fehlende Datei → leere Liste).
- **Sound entkoppelt & non-blocking.** `Game` ruft nur `sounds.play(name)`; in Tests
  wird ein `NullSound` injiziert. WAVs werden synthetisiert (keine Asset-Dateien),
  Wiedergabe via `afplay` ohne die Game-Loop zu blockieren.
- **Logging statt Prints.** Rotierendes Logfile unter `logs/`; unbehandelte Fehler
  werden mit Traceback geloggt → nachvollziehbares Debugging nach einem Problem.
- **Test-case-basiert.** Die gesamte Logik (Geometrie, Tracking, Physik, Spiel) ist
  headless über `pytest` abgesichert und unabhängig von Kamera/GUI testbar.

## Qualitätssicherung
- **Tests:** `fruit-ninja-cv/tests/` (pytest), laufen ohne Kamera/GUI.
- **Logging:** `logs/fruit-ninja.log` (rotierend), Konsole zeigt nur Warnungen/Fehler.
- **CI:** GitHub Actions (`.github/workflows/tests.yml`) führt bei jedem Push und
  Pull Request alle Tests aus. In `main` kommt nur Code mit grünen Tests.

## Prozessentscheidungen
- **Noahs Programm ist die gemeinsame Basis.** Es gab zwei parallele Umsetzungen
  (Branches `FruitninjaLeon` und `FruitninjaNoah`). Das Team hat sich für Noahs
  Projekt entschieden (modularer Aufbau, mehr Funktionen, umfangreichere Tests).
  `FruitninjaNoah` wurde mit seiner Historie in `main` übernommen; Leons Stand bleibt
  im Branch `FruitninjaLeon` erhalten und kann als Ideenquelle dienen (z. B.
  Kalibrierung per Farbhistogramm, Schnitt erst ab Mindestgeschwindigkeit,
  Kamera-Setup für die ELP-USB-Kamera unter Windows).
- **Maßgeblich ist `aufgabenstellung.md`.** Noahs frühere `Aufgabenstellung.md`
  wurde nicht übernommen (veraltet; kollidiert unter Windows wegen gleicher
  Schreibweise bis auf Groß-/Kleinschreibung).
- **Iterationen über GitHub.** Jede Iteration ist ein Milestone, Aufgaben sind
  Issues, ein Branch pro Issue, Änderungen nur per Pull Request mit Tests.

## Offen gegenüber der Aufgabenstellung
- **Netzwerk-Mehrspieler (Teilaufgabe 6)** – fehlt noch, Architektur offen.
- **Schnittgeste (Teilaufgabe 4):** Aktuell schneidet schon Berührung ohne
  Mindestgeschwindigkeit; gefordert ist, dass langsame/ruhende Bewegungen kein
  Schnitt sind.
- **Tracking (Teilaufgabe 3):** Glättung von Messrauschen und Überbrücken kurzer
  Erkennungsaussetzer.
- **Stiftspitze (Teilaufgabe 2):** Aktuell wird der Schwerpunkt verwendet.
- **Debug-Fenster** mit Kamerabild und erkannter Stiftposition (bisher Maske per `d`).
- **Windows:** Standard-Kamera (`CAMERA_INDEX`) und Sound (`afplay`) sind auf macOS
  ausgelegt; unter Windows mit der ELP-USB-Kamera prüfen.

## Bewusste Nicht-Ziele
- Kein ML für die Stifterkennung (klassische Bildverarbeitung). Ausnahme:
  Spielererkennung per Gesicht (siehe oben).
