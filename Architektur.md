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
| `main.py` | Game-Loop, State-Machine, Kamera-Capture, Tastatur/Maus, Rendering-Orchestrierung, Logging-Setup, Fehlerbehandlung |
| `config.py` | Alle Konstanten (Kamera, HSV, Physik, HUD) an einer Stelle |
| `marker_tracker.py` | HSV-Farb-Segmentierung + Kalibrierung + Positionsbestimmung |
| `blade.py` | Kurzer Trail der letzten Marker-Punkte; aktuelles Segment & Geschwindigkeit |
| `fruit.py` | Frucht-/Bomben-Entität: Projektil-Physik + prozedurale Darstellung |
| `game.py` | Spielzustand: Spawning (inkl. Bursts), Kollision, Punkte, Leben |
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
- **HUE-basierte Marker-Erkennung.** Der Farbton ist das Hauptkriterium; Sättigung
  und Helligkeit haben nur einen Mindestwert (`S_FLOOR`/`V_FLOOR`). Robust gegen
  Lichtschwankungen und gegen den hellen Hintergrund (weiße Wand). Kalibrierung per
  Klick nimmt den Median-Hue eines kleinen Flecks (präziser als Box-Mittelwert).
- **Pinker Marker.** Farbton liegt weit vom Hautton entfernt → keine Verwechslung
  mit der Hand.
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
- **Logging statt Prints.** Rotierendes Logfile unter `logs/`; unbehandelte Fehler
  werden mit Traceback geloggt → nachvollziehbares Debugging nach einem Problem.
- **Test-case-basiert.** Die gesamte Logik (Geometrie, Tracking, Physik, Spiel) ist
  headless über `pytest` abgesichert und unabhängig von Kamera/GUI testbar.

## Qualitätssicherung
- **Tests:** `fruit-ninja-cv/tests/` (pytest), laufen ohne Kamera/GUI.
- **Logging:** `logs/fruit-ninja.log` (rotierend), Konsole zeigt nur Warnungen/Fehler.

## Bewusste Nicht-Ziele
- Kein ML, kein Netzwerk, keine persistente Highscore-Speicherung (optional später).
