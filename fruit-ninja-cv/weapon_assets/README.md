# Generiertes Laserschwert

`laser-sword.png` wurde mit dem eingebauten `image_gen` einmalig am 26.11.2026
erzeugt: geradlinige cyanfarbene Energieklinge und ein dunkler Comic-Metallgriff
mit silbernen Ringen und leuchtenden Akzenten. Vollständiger Prompt, SHA-256-Werte
und Pixelanker stehen in `laser-sword.json`. Das generierte RGBA wurde unter
Erhalt der Transparenz auf 512×768 verkleinert; es wird lokal geladen.

Die Renderer teilen die Grafik am Emitter in Klinge und Griff und passen beide
mit getrennten Affintransformationen an die vorhandene Waffenpose an. Der
generierte äußere Glanz wird beim Laden weich auf die Umgebung der Klinge
begrenzt; der bestehende dynamische Halo ergänzt die aktuelle Stiftfarbe. Der
weiße Kern und die Klingen-Spitze bleiben gut lesbar. Spitze und Griff folgen
unverändert dem bisherigen Tracking. Die Trefferauswertung verwendet weiterhin
den Stiftpunkt, nicht den Lichtschein oder die neue Bildfläche.

Einmaliger Ladecache, ROI-Compositing und vorhandene OpenCV-Abhängigkeit; kein
Netzwerk oder Bildgenerator zur Laufzeit. Bei fehlender Grafik wird die
prozedurale Waffe mit einmaligem Logeintrag gezeichnet.

`python tools/preview_comic_art.py` im Programmordner erzeugt Galerie, Schnitte,
Schwertrotationen und Messwerte ohne Webcam. Eine neue Live-Abnahme wurde nicht
durchgeführt.
