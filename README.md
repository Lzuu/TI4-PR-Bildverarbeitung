# TI4 – PR Bildverarbeitung: Fruit Ninja mit Farberkennung

Projekt im Modul **Bildverarbeitung** (TI, Semester 4 – HS Albstadt-Sigmaringen).

> Basis ist Noahs Programm aus dem Branch `FruitninjaNoah` (Teamentscheidung).
> Es wird von hier aus gemeinsam weiterentwickelt.

## Projektbeschreibung
Ein **Fruit-Ninja-Klon**, der mit einem **farbigen Stift als Schwert** (Pink/Gelb/
Grün) vor einer Webcam gesteuert wird – ohne Maus, Tastatur oder Touch. Früchte
werden von unten ins Bild geworfen (Wurf-Physik) und fliegen im Bogen auf; der/die
Spieler:in zerschneidet sie, indem der Stift die Frucht **berührt**. Der Stift wird
live per **OpenCV** anhand seiner Farbe aus dem Kamerabild erkannt und verfolgt.

## Unsere Idee
Statt klassischer Eingabegeräte nutzen wir das **Kamerabild als Controller**:
- Der **farbige Stift** wird über seine **Farbe (HSV)** vom Hintergrund getrennt
  (ideal vor einer weißen Wand) und seine Position in Echtzeit getrackt.
- Auswahl (Farbe & Schwierigkeit) **VR-artig per Dwell**: Stift 3 s in einen Kreis
  halten, ein Ladering bestätigt.
- Die Position bildet eine **Klinge**; berührt sie eine Frucht, wird diese geschnitten.
- **Bomben** dürfen nicht getroffen werden – das sorgt für Spannung.

Das Spiel ist bewusst klein gehalten, deckt aber eine komplette
Bildverarbeitungs-Pipeline ab.

## Dokumentation
- **[aufgabenstellung.md](aufgabenstellung.md)** – offizielle Aufgabenstellung und Anforderungen.
- **[Architektur.md](Architektur.md)** – Aufbau, Module, Architekturentscheidungen.
- **[fruit-ninja-cv/README.md](fruit-ninja-cv/README.md)** – Start, Steuerung, Tuning, Tests.

## Bildverarbeitungs-Pipeline (Kern)
1. **Capture & Preprocessing** – Frame holen, skalieren, spiegeln.
2. **Segmentierung** – Stift-Farbe per HSV-Schwellwert (`cv2.inRange`), Morphologie.
3. **Tracking** – größte Kontur, Schwerpunkt des Stifts bestimmen.
4. **Physik & Kollision** – Früchte als Projektile; Treffer über Abstand zur Klinge.
5. **Rendering** – Früchte, Klingen-Trail und HUD ins Bild zeichnen.

## Qualität & Prozess
- **Test-case-basiert:** automatisierte Tests unter `fruit-ninja-cv/tests/` (pytest).
- **Logfiles:** Laufzeit-Ereignisse & Fehler in `logs/fruit-ninja.log` (rotierend).
- **Inkrementell-iterativ:** Zuwächse werden als **GitHub-Issues** abgebildet.
- **Automatische Tests:** GitHub Actions führt bei jedem Push und Pull Request alle
  Tests aus (`.github/workflows/tests.yml`).

## Arbeitsweise
1. **Iterationen** sind [Milestones](../../milestones) auf GitHub, Aufgaben sind Issues darin.
2. **Ein Branch pro Issue**, von `main` abzweigen: `git switch -c <nr>-<kurzname>`.
3. **Test zuerst:** Testcase in `fruit-ninja-cv/tests/` schreiben, dann implementieren.
   Jede Änderung bekommt neue oder angepasste Tests.
4. **Alle Tests ausführen** (`cd fruit-ninja-cv` und `python -m pytest`), erst dann Pull Request.
5. **Pull Request gegen `main`** mit `Closes #<nr>`. Gemergt wird nur, wenn alle Tests grün sind.
6. **Architekturentscheidungen** in [Architektur.md](Architektur.md) ergänzen.

## Schnellstart
```bash
cd fruit-ninja-cv
pip install -r requirements.txt
python main.py            # Spiel starten
pytest                    # Tests ausführen (requirements-dev.txt)
```

## Tech-Stack
`Python` · `OpenCV` · `NumPy` · `pytest`
