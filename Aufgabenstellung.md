# Aufgabenstellung

Modul **Bildverarbeitung** · TI, Semester 4 · HS Albstadt-Sigmaringen

## Ziel (grob)
Eine interaktive Anwendung, die mit einer **Webcam** und **OpenCV (Python)** ein
Live-Kamerabild in Echtzeit auswertet und daraus eine Steuerung ableitet. Umgesetzt
als **Fruit-Ninja-Klon**: Ein **pinker Stift** dient als „Schwert"; Früchte werden
von unten ins Bild geworfen und müssen durch Berührung mit dem Stift zerschnitten
werden.

## Idee
Das **Kamerabild wird zum Controller** – keine Maus/Tastatur im Spiel. Der pinke
Stift wird farbbasiert segmentiert und verfolgt; seine Position bewegt die Klinge.
Berührt die Klinge eine Frucht, wird sie geschnitten. Bomben dürfen nicht berührt
werden.

## Anforderungen

### Funktional
- Live-Bild der Webcam erfassen und anzeigen.
- Pinken Stift farbbasiert erkennen und seine Position in Echtzeit verfolgen.
- Kalibrierung der Stiftfarbe (Klick auf den Stift oder Box + SPACE).
- Früchte mit **Wurf-Physik** von unten einwerfen (Parabelflug).
- Schneiden durch **Berührung** (keine Mindestgeschwindigkeit nötig).
- Punkte, Leben und Game-Over-Logik; Neustart.
- Bomben als Negativ-Objekte (Treffer = Game Over).

### Technisch / Rahmenbedingungen
- Sprache & Bibliothek: **Python + OpenCV** (klassische Bildverarbeitung).
- Hardware: **Webcam** (MacBook FaceTime HD), einfacher Hintergrund (weiße Wand),
  pinker Stift als Marker.
- Echtzeitfähig (flüssige Bildrate).
- Bewusst **kleiner Projektumfang**.

### Qualität / Prozess
- **Test-case-basiert**: Logik durch automatisierte Tests (pytest) abgesichert.
- **Logfiles** zum Debugging: Laufzeit-Ereignisse und Fehler werden persistent
  protokolliert (`logs/`).
- **Inkrementell-iterative** Entwicklung: Zuwächse werden als **GitHub-Issues**
  abgebildet und schrittweise umgesetzt.

## Abgrenzung / Scope
- Kein Machine Learning / keine fertigen Hand-Tracking-Libs – nur klassische
  OpenCV-Verfahren (Farb-Segmentierung, Morphologie, Konturen).
- Keine Netzwerk-/Online-Funktionen, kein persistenter Highscore (optional später).

## Ergebnis / Deliverables
- Lauffähiges Spiel (`fruit-ninja-cv/`).
- Automatisierte Testfälle (`fruit-ninja-cv/tests/`).
- Dokumentation: diese Datei, `Architektur.md`, READMEs.
