# Generiertes Laserschwert

`laser-sword.png` wurde mit dem eingebauten `image_gen` einmalig am 26.11.2026
erzeugt: geradlinige cyanfarbene Energieklinge und ein dunkler Comic-Metallgriff
mit silbernen Ringen und leuchtenden Akzenten. Vollständiger Prompt, SHA-256-Werte
und Pixelanker stehen in `laser-sword.json`. Das generierte RGBA wurde unter
Erhalt der Transparenz auf 512×768 verkleinert; es wird lokal geladen.

Seit #89 wird nur die Materialgrafik des Griffs verwendet. Der eingebrannte
Lichtsaum der Bildklinge wurde beim Strecken als rechteckige Farbfläche sichtbar.
Die Energie entsteht jetzt direkt entlang der aktuellen Klingenachse aus einem
einzigen weichen Halo, cyanfarbener Fassung und weißem Kern. `SABER_COLOR` ist von
der Erkennungsfarbe getrennt. Der originale PNG und seine Prüfsumme bleiben
unverändert; beim einmaligen Laden des Griffs werden transparente Lichtreste um
die Materialkontur unterdrückt.

Die Griffgrafik folgt weiter Emitter→Knauf. Eine schmale, abgerundete Überdeckung
aus aktuellen Hand-/Handschuhpixeln und ein lokaler Kontaktschatten betten den
Griff ein. Das ist eine visuelle Näherung ohne Fingertracking. Die zeitliche
Stabilisierung der Waffenpose ist in `lightsaber.py` dokumentiert. Treffer nutzen
weiterhin ausschließlich gültige aktuelle Stiftpunkte.

Einmaliger Ladecache, ROI-Compositing und vorhandene OpenCV-Abhängigkeit; kein
Netzwerk oder Bildgenerator zur Laufzeit. Bei fehlender Grafik wird die
prozedurale Waffe mit einmaligem Logeintrag gezeichnet.

`python tools/preview_comic_art.py` im Programmordner erzeugt Galerie, Schnitte,
Schwertrotationen und Messwerte ohne Webcam. Eine neue Live-Abnahme wurde nicht
durchgeführt.

`python tools/preview_saber_stability.py` ergänzt Vorher/Nachher und einen
synthetischen Bewegungs-Replay unter `docs/stable-saber-89/`.
