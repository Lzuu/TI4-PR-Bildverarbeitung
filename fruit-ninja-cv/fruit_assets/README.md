# Fruchtgrafiken

Einmalig am 26.11.2026 mit dem eingebauten `image_gen` erzeugte Grafiken:
Apfel, Orange, Limette, Blaubeere, Zitrone und eine stilisierte 3D-Spielbombe.
Kein Bildgenerator, Netzwerkzugriff oder neues Paket ist zur Laufzeit erforderlich.
Der erste realistische Bombenentwurf wurde vom Generator abgelehnt; verwendet
wird die anschließend erfolgreich erzeugte, klar fiktive Spielgrafik.

`generation.json` enthält die vollständigen finalen Prompts und SHA-256-Prüfsummen.
Die generierten RGBA-Bilder wurden unter Beibehaltung ihrer Alpha-Kanten skaliert
und auf einheitliche Körperanker gesetzt. Alpha-Quantisierungsreste von höchstens
2/255 im leeren Hintergrund wurden entfernt. Früchte: 320×320, Körperzentrum
(160,184), Radius 120. Bombe: 400×400, Zentrum (200,260), Radius 120; der zusätzliche
obere Platz gehört zur Lunte. Trefferprüfung verwendet weiterhin den bisherigen
Fruchtradius, ohne dekorative Stiele/Lunte einzubeziehen.

Die Schnittanimation teilt die vorhandene Außenansicht an der aktuellen
Klingenrichtung; separate Innenansichten von Fruchtfleisch sind nicht enthalten.

Reproduzierbare Vorschau und Messung (ohne Webcam), im Programmordner:

```powershell
python tools/preview_fruits.py
```

Die sechs finalen PNGs werden direkt aus diesem Ordner geladen. Fehlt eine Datei,
erscheint die bisherige prozedurale Ersatzgrafik mit einmaligem Logeintrag.
