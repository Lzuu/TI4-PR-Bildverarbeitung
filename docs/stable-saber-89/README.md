# Laserschwert: Farbe, Handkontakt und Stabilität (#89)

## Behobene Ursachen

- Der alte Spielaufruf übergab die Erkennungsfarbe Pink als Halo- und Spurfarbe,
  obwohl die Bildklinge bereits cyan war. Beide Lichtfarben lagen übereinander.
- Der in der Bitmap enthaltene breite Glow wurde mitgestreckt und erzeugte vor
  allem bei kurzen Klingen eine rechteckige Farbfläche. Die Bilddatei ist
  unverändert; verwendet wird jetzt nur ihre bereinigte Griffkontur. Energie
  wird kontinuierlich auf der Waffenachse gerendert.
- Jeder einzelne Handschuhpunkt wurde ungefiltert übernommen. Nach nur einem
  fehlenden Handschuh-Frame wechselte der Griff zur anders arbeitenden Hautsuche.
  Eine fehlende Hand führte zu selbständiger Drehung in die Senkrechte, ein
  fehlender Stift setzte die ganze Waffe sofort zurück. Diese Wechsel flackerten.
- Der zuletzt vollständig vor die Hand gezeichnete Griff hatte keine Überdeckung.

## Umsetzung

Die Visualisierung bekommt `f.now` und Kamera/Modus/Auflösung/Stiftfarbe als
Quellkennung. Kleine Positionsfehler werden mit zeitabhängiger Glättung reduziert;
der sichtbare Spitzenfehler ist auf 3 Pixel × SCALE plus Rundung begrenzt. Bewegung
über 12 Pixel × SCALE pro Sample passiert ohne Tip-Verzögerung. Die relative
Griffposition wird geglättet und höchstens 5 Pixel × SCALE von der aktuellen
Messung entfernt gehalten. Große unkoordinierte Griffsprünge brauchen 50 ms
konsistente Beobachtung; entsprechend große gemeinsame Pen-/Handbewegungen werden
nicht gebremst. Haut-Ausreißer müssen ebenfalls dieselbe Richtung bestätigen.

Die Hand bleibt gebunden, bis eine andere Hand 100 ms deutlich näher ist. Eine
Pose-Lücke hält höchstens 150 ms die letzte relative Waffenform, übersetzt mit
dem aktuell beobachteten Stift, nicht mit alten absoluten Handschuhkoordinaten.
Danach greift die bestehende Hautsuche. Ohne Hand dreht die Waffe nicht von selbst.
Ein fast zur Kamera zeigender Stift liefert keine belastbare Achse: dort bleibt
die letzte Orientierung erhalten und eine hinter dem Emitter liegende Klinge
wird nicht als umgekehrter Strahl gezeichnet.

Bei fehlendem Marker bleibt nur die Waffendarstellung maximal 120 ms stehen und
verblasst. Gameplay und Challenge sehen weiterhin `None`: keine Schnitte mit
gehaltenen Punkten, kein Sprungschnitt bei Wiedererkennung. Kamera-/Modus- und
Farbwechsel verwerfen die visuelle Historie. Ohne übergebene Zeit wird für
Standalone-Aufrufe die REF_FPS-Schrittweite verwendet.

Eine aktuelle Handschuh-Alpha-Maske oder die vorhandene Hautmaske überdeckt schmale,
abgerundete Bänder des Griffs. Ein weicher lokaler Kontaktschatten ergänzt die
Überlagerung. Dieses konservative Fingerbild ist **keine** geschätzte 3D-Fingerpose
und kann die Haltung einer geöffneten Hand nicht in eine echte Faust verwandeln.
Unsichere/versetzte Handschuhmasken werden nicht wiederverwendet. Die Original-
Kamerabilder für Erkennung werden nicht verändert. Keine neuen Modelle oder
Laufzeit-Abhängigkeiten.

## Reproduzierbare Überprüfung

Im Ordner `fruit-ninja-cv`:

```powershell
python -m pytest
python tools/preview_saber_stability.py
```

`contact-before-after.png`: derselbe synthetische Spieler, alter Renderpfad aus
Git-Stand `6a7e87d` neben aktueller Darstellung. Die Szene verwendet das bereits
freigegebene öffentliche NASA-Testportrait. Ohne Git-Historie wird nur die neue
Ansicht erzeugt. `motion-cases.png` und `motion-replay.mp4` zeigen Ruhe-Zittern,
einen Griffausreißer, Pose-/Marker-Lücken und schnelle Translation. GIF-Export ist
zusätzlich möglich, wenn Pillow lokal vorhanden ist; OpenCV-Preview und Video
benötigen Pillow nicht.

Gemessene synthetische Ruhe-Sequenz, vier OpenCV-Threads: Summe der
Standardabweichungen in X/Y am Griff vorher 4,855 Pixel, nachher 0,795 Pixel;
Spitze 4,855 → 1,463 Pixel. Tracker-Median 0,067 ms, Zeichnen inklusive aktueller
Handüberdeckung Median 0,594 ms / P95 0,817 ms (60 warme Aufrufe bei 960×720).
Rohwerte in `measurements.json`; keine Live-FPS- oder Genauigkeitsgarantie.

Tests decken Zeitverhalten bei 15/30/60 Hz, Jitter, schnelle kohärente Bewegung,
Ausreißer, Handzuordnung, Pose-/Marker-Lücken, Quellenwechsel, ungültige Zeit,
Handüberdeckung, Alpha/Bildränder, Material-Fallback und unabhängige Lichtfarbe
im echten GameLoop ab. Die bestehenden Tests wurden dort aktualisiert, wo sie
das zuvor ausdrücklich gewünschte sofortige Umspringen oder Rückdrehen verlangten.

Keine Webcam/GUI gestartet. Noch live prüfen: schnelle Schwünge und Drehungen,
geöffnete/geschlossene Hände, schlechte Beleuchtung, seitliche Kameraabstände und
wiederholte Verdeckung. Die Qualität der vorhandenen Marker-/Pose-Erkennung bleibt
die Grenze; lange Ausfälle werden nicht durch eine schwebende Waffe kaschiert.
