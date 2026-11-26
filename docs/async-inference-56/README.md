# Issue #56: Inferenz außerhalb der Spielschleife

Ein gemeinsamer Worker serialisiert YOLO und Pose. Die Oberfläche übergibt Kopien
des Originalbilds und bei Pose auch der Maske. Je Typ wartet maximal ein Auftrag;
ein neuer ersetzt den wartenden Auftrag. Es gibt keine anwachsende Bildschlange.
Modelle laden/verarbeiten ausschließlich im Worker. Fehler deaktivieren den
betroffenen Typ einmalig, andere Erkennung und Spiel bleiben nutzbar. YOLO lässt
sich durch erneute Moduswahl nach einem Fehler neu laden.

Ergebnisse sind an Quelle (Kamera, Modus-Sitzung), Auflösung und Generation
gebunden. Auch Kamera A → B → A kann kein noch laufendes altes A-Ergebnis
wiederverwenden. Dasselbe gilt für Fehler alter Aufträge. Shutdown verwirft die
Postfächer und wartet nicht auf einen möglicherweise blockierenden nativen Aufruf.
Der Daemon beendet sich nach dessen Rückkehr; Python kann ihn nicht sicher mitten
in `forward()` abbrechen. Er hält keine Kameras oder Fenster offen.

Das Ergebnisalter zählt ab dem UI-Zeitstempel unmittelbar nach dem Lesen des
Kamerabilds, einschließlich Warteschlange und Verarbeitung. Die Webcam selbst
liefert keinen Hardware-Aufnahmezeitstempel; ältere Treiberpuffer werden hierdurch
nicht erfasst. Pose-Landmarken werden mit optischem Fluss vom tatsächlich
inferierten Bild auf das aktuelle Bild übertragen. Fluss verlängert das
Verifizierungsalter nicht. Die Pose verwendet Landmarken 33/34 des Zoo-Modells
für den nächsten Ausschnitt; der Personendetektor läuft nur nach Verlust oder
Quellen-/Körperwechsel erneut.

## Altersbudgets und Grenzen

Für asynchrone Masken gilt ein explizites Budget von 300 ms. Die bisherigen
100 ms waren ein Aktualisierungsziel des synchronen Segmentierers; dessen
Rückgabewert konnte bei langsamer Inferenz bereits älter sein. Als harte Grenze
im asynchronen Pfad verworfen diese 100 ms hier sämtliche YOLO-Ergebnisse.
Das größere, weiterhin begrenzte Budget erlaubt Ergebnisse langsamer Hardware,
vergrößert aber die mögliche Silhouettenverzögerung. Pose bleibt auf 200 ms
begrenzt. Bei Überschreiten verschwindet der betreffende Overlay-Pfad; es wird
kein beliebig altes Ergebnis gezeigt. Auf sehr langsamen Geräten kann die
Erkennung daher zeitweise fehlen. Bewegung und Beleuchtung müssen live geprüft
werden, insbesondere schnelle seitliche Bewegungen und verdeckte Arme.

OpenCV ist vor Workerstart prozessweit auf vier Threads begrenzt; dadurch laufen
auch die übrigen OpenCV-Schritte mit diesem Limit. Das ist über config.py
konfigurierbar. Vier ist eine gemessene Ausgangseinstellung für dieses Gerät,
keine universelle optimale Threadzahl.

## Reproduzierbare Messung

Im Ordner `fruit-ninja-cv`, mit bereits lokal installierten ONNX-Modellen:

```powershell
python tools/benchmark_async_inference.py --frames 100
python -m pytest
```

`measurements.json`: OpenCV 4.13.0, vier Threads, öffentliches statisches
NASA-Portrait 480×600, 15 Warm-up-Frames und 100 Messframes je Modus mit Zielabstand
1/30 s. Gemessen wird nur der Anteil für Segmentierung und Pose auf dem UI-Thread.
Kamera, Helm/Anzug-Rendering, übrige Spiellogik und Anzeige sind nicht enthalten.

| Messgröße | Synchron | Asynchron |
| --- | ---: | ---: |
| Median UI-Arbeit | 2,146 ms | 2,053 ms |
| P95 UI-Arbeit | 115,453 ms | 2,787 ms |
| Maximum UI-Arbeit | 119,903 ms | 3,639 ms |
| Frames mit Maske | 100/100 | 100/100 |
| Frames mit Pose | 90/100 | 78/100 |
| Größtes beobachtetes Maskenalter | 153,736 ms | 258,570 ms |

Die reduzierte Pose-Verfügbarkeit wird ausdrücklich mit ausgewiesen: Der
asynchrone Pfad verwendet strikte Aufnahmezeit-Altersgrenzen; das Portrait zeigt
keinen ganzen Körper und das Modell verliert die Pose wiederholt. Flüssigere
UI-Arbeit bedeutet hier weder bessere Erkennungsgenauigkeit noch 30 Live-FPS.
Der Vergleich nutzt bereits in beiden Pfaden die neue Landmarken-ROI-Wiederverwendung.

## Tests

389 bestanden, 17 Bash-Tests unter Windows übersprungen. Neue Tests verwenden
Ereignisse statt unzuverlässiger Sleep-Schwellen und prüfen einen blockierten
Fake-YOLO-Aufruf durch die echte Spielschleife: Zeichnen und Beenden funktionieren
noch vor Freigabe des Netzes. Weitere Fälle: neuestes wartendes Bild, Kopien der
Eingaben, exklusiver Worker, Quellenwechsel einschließlich A/B/A, Fehler alter
Quellen, einmaliges Abschalten, Masken-/Posealter, Zuordnung des Flussbilds,
Modellladen im Worker, Personendetektor nach Verlust und Diagnosewerte im echten
Loop. Die bisherigen Integrationstests bleiben als deterministischer synchroner
Vergleichspfad erhalten; die neuen Integrationstests aktivieren den Produktionspfad.

Keine echte Webcam oder GUI gestartet; Live-Abnahme am Demo-Laptop steht aus.
