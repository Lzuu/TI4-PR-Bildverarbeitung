# Lokale EVA-Anzugteile

Acht getrennte RGBA-PNGs, einmalig mit dem eingebauten `image_gen`-Werkzeug
(Imagegen-Skill) während der Entwicklung am 29.10.2026 erzeugt. Keine Person und
kein Gesicht sind Bestandteil der Assets. `helmet.png` enthält eine echte
transparente Visieröffnung. Die Kamera und das Glas werden erst im Spiel eingesetzt.

| PNG | Bewegliche Verankerung |
| --- | --- |
| torso.png | Vier Punkte: beide Schultern und beide Hüften; eigener Halsring/Brustblock |
| upper_arm.png | Schulter → Ellbogen |
| forearm.png | Ellbogen → Handgelenk |
| glove.png | Handgelenk → Finger-/Handrichtung |
| thigh.png | Hüfte → Knie |
| shin.png | Knie → Knöchel |
| boot.png | Knöchel → Fußspitze |
| helmet.png | Gesicht/Augen, separate Kameraöffnung und Glasschicht |

`anchors.json` speichert normierte Anker. Die tatsächliche Alpha-Breite wird beim
Laden vermessen, damit transparente Bildränder die Proportionen nicht verfälschen.
Die Öffnung der Helmschale wird aus der eingeschlossenen transparenten Region
ermittelt. Laufzeitgröße maximal 384px pro PNG, gemeinsamer begrenzter Cache.
Die Original-PNGs bleiben in voller erzeugter Auflösung erhalten.

## Finale Prompts und Herkunft

Gemeinsame Vorgabe für die Erzeugung: photorealistic classical EVA garment,
warm ivory/off-white fine woven Beta cloth, realistic sewn seams, folds and
padding, neutral grey hardware, soft studio light from upper left,
orthographic FRONT view, isolated modular component, genuinely transparent PNG,
no checkerboard, no backdrop, no person, no skin, no face, no mannequin, no logo
or text. Centered component, square canvas, no external cast shadow.

Die vollständigen Komponenten-Prompts und die gezielte Helm-Überarbeitung stehen
in [prompts.json](prompts.json). `checksums.json` erlaubt die unveränderten
ausgelieferten Pixel zu verifizieren. Generierung ist nicht deterministisch;
reproduzierbare Vorschauen verwenden die gespeicherten Dateien, nicht neue Bilder.
