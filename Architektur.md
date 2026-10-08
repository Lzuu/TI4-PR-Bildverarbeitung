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
| `main.py` | Game-Loop, State-Machine (Menu/Difficulty/Play/GameOver), Kamera-Capture, Dwell-Auswahl, Maus-Farbeinstellung, Rendering, Logging-Setup, Fehlerbehandlung |
| `config.py` | Gemeinsame Konstanten (Kamera, Marker, Dwell, Kombos, Highscore, HUD) |
| `colors.py` | Feste Stiftfarben Pink/Gelb/Grün (HSV-Bereich + UI-Farbe) |
| `difficulty.py` | Schwierigkeitsgrade (Tempo; immer 3 Leben) als Presets |
| `dwell.py` | VR-Dwell-Auswahl (Stift n Sekunden auf einem Ziel = Bestätigung) |
| `marker_tracker.py` | HSV-Farb-Segmentierung (eine oder mehrere Farben) + Positionsbestimmung |
| `blade.py` | Kurzer Trail der letzten Marker-Punkte; aktuelles Segment & Geschwindigkeit |
| `fruit.py` | Frucht-/Bomben-Entität: Projektil-Physik + prozedurale Darstellung |
| `game.py` | Spielzustand: Spawning (Bursts), Kollision, Punkte, Leben, Kombos |
| `heart.py` | Herz-Sprite (parametrische Herzkurve, Verlauf+Glanz) für die Leben |
| `highscore.py` | Top-3-Highscore **je Schwierigkeit** (JSON-Persistenz) |
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
  ermittelt (`tkinter`) und als Render-Canvas gesetzt; das Fenster läuft im Vollbild
  (kein grauer Rand). Ein globaler Faktor `config.SCALE = HEIGHT / REF_HEIGHT`
  skaliert UI (`utils.draw_text`, Kreise) **und Physik** (Radius, Wurf, Gravitation
  in `fruit.py`). Radius und Geschwindigkeit/Gravitation werden gemeinsam skaliert,
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

## Bewusste Nicht-Ziele
- Kein ML, kein Netzwerk, keine persistente Highscore-Speicherung (optional später).
