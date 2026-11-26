# Anzug-Zeitmessung (#62)

Start wie bisher im Ordner `fruit-ninja-cv`: `python main.py`. Taste `p` schaltet
das HUD ein. Der wirkungslose Schalter `--suit-diagnostics` ist entfernt; es gibt
keinen zweiten Diagnosepfad. Die bereits entfernten `last_ms`-Felder bleiben weg.

| Unterstufe | Gemessene Arbeit |
| --- | --- |
| `suit_masks` | Hauptregion auswählen, Stoffregion und Clip-Masken vorbereiten |
| `suit_fabric` | ROI-Kopie, Stoffstruktur/Shading und Einsetzen des Untergrunds |
| `suit_rig` | Kopfanker, Körpergröße, Teile und Transformationsmatrizen |
| `suit_sprites` | Alpha-Compositing vor und nach dem Helm, je Frame summiert |
| `suit_helmet` | Originalgesicht, Helmschale und Visier/Reflexion |
| `suit_restore` | Unveränderte Kamera-/Stiftpixel zurückkopieren |

Stoffinterne Shading-Masken gehören zur Stoffzeit; die Maskenzeile misst die
Maskenvorbereitung des Rigs. Kopfanker gehören zur Rig-Zeit, nicht zur Helm-
Zeichenzeit. Die äußere Zeile `suit` enthält alle Unterstufen einschließlich
Steuerungs- und Messaufwand. Ihre Zeit **nicht nochmals zu den Unterstufen addieren**.

Wiederholte Pässe liefern ein summiertes Sample je Unterstufe/Frame. Deshalb zählt
`n` Frames, nicht Sprite-Pässe oder einzelne Körperteile. Nur ausgeführte Stufen
werden veröffentlicht: Ohne Spieler gibt es nur Maskenzeit, beim Rückfall auf das
Originalbild nur Masken/Rig/Wiederherstellung und ohne Kopf keine Helmzeit. Ein
Fenstermedian bezieht sich auf die Frames, in denen die Stufe tatsächlich lief.

Die vorhandenen Logs schreiben Median/P95 im `PERF_LOG_INTERVAL`. Optionaler
CSV-Export über `PERF_CSV` in config.py enthält dieselben Stufennamen und Counts.
Das HUD hält `suit` und seine Unterstufen zusammen innerhalb seiner zwölf Zeilen.
Es gibt weder Pro-Frame-Logging noch globale oder threadübergreifende Zeitpuffer.

## Reproduzierbare Prüfung und Messung

```powershell
python -m pytest
python tools/benchmark_suit_stages.py --iterations 60 --threads 4
```

Der Benchmark benötigt keine Modelle, Kamera oder Fenster. Er rendert die
synthetische seitliche Armhaltung bei 960×720, acht Warm-up-Aufrufe und 60 Paare
mit abwechselnder Reihenfolge mit/ohne Empfänger. Kopieren des Eingabebilds ist
außerhalb der gemessenen Renderzeit. Threadzahl ist explizit protokolliert.

Messlauf OpenCV 4.13.0, vier Threads:

| Größe | Median | P95 |
| --- | ---: | ---: |
| Renderer ohne Empfänger | 24,215 ms | 26,101 ms |
| Renderer mit Messung | 24,257 ms | 26,704 ms |
| Masken | 6,343 ms | 7,384 ms |
| Stoff | 3,680 ms | 4,280 ms |
| Rig | 1,727 ms | 2,333 ms |
| Sprites | 6,848 ms | 7,568 ms |
| Helm | 4,711 ms | 5,515 ms |
| Wiederherstellung | 0,069 ms | 0,095 ms |

Die Differenz der Gesamtmediane beträgt hier 0,042 ms; das ist eine einzelne
Headless-Messung unter Systemlast, keine garantierte Obergrenze des Messaufwands.
Die Mediane/P95 der Unterstufen sind eigenständige Verteilungen und summieren sich
nicht exakt zum Median/P95 der Gesamtzeit. Keine Aussage über Live-FPS oder
Kamera-zu-Anzeige-Latenz. Der separate opt-in-Zeitbudgettest aus #57 wird vom
normalen Testaufruf ausdrücklich nicht ausgeführt (`-m perf` nur auf Abruf).

Neue Tests prüfen deterministisch summierte Zeiten, Fehlerweitergabe, optionalen
taktfreien Pfad, identische Bildpixel, fehlenden Spieler/Tracking-Rückfall,
wirkliche Log-/CSV-Ausgabe aus dem GameLoop und Sichtbarkeit aller Unterstufen im
begrenzten HUD. Der CLI-Test ruft `--help` auf und prüft, dass der entfernte
Schalter vor dem Spielstart als ungültig abgelehnt wird. Ein zusätzlicher AST-
Vergleich gegen den Basisstand bestätigt die unveränderte Render-Operationsfolge
nach Entfernen der Timing-Kontexte. Keine Webcam/GUI gestartet.
