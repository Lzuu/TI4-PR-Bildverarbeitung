# Aufgabenstellung: Fruit Ninja mit Stift-Tracking

## Ziel

Wir entwickeln eine Bildverarbeitungsanwendung, die einen Stift über eine USB-Kamera in Echtzeit erkennt und verfolgt. Der Stift ersetzt dabei Maus und Touchscreen: Seine Bewegungen steuern das Spiel **Fruit Ninja**. Wer mit dem Stift vor der Kamera durch die Luft wischt, zerschneidet im Spiel die Früchte.

## Teilaufgaben

### 1. Bilderfassung
- Kamerabild einer USB-Kamera fortlaufend einlesen
- Bildrate und Auflösung so wählen, dass das Spiel flüssig reagiert

### 2. Stifterkennung
- Den Stift in jedem Kamerabild finden, z. B. über seine Farbe, seine Form oder eine Markierung an der Spitze
- Die Position der Stiftspitze als Bildkoordinate bestimmen
- Die Erkennung soll bei wechselndem Licht und unruhigem Hintergrund zuverlässig bleiben

### 3. Stiftverfolgung (Tracking)
- Die Position des Stifts über mehrere Bilder hinweg verfolgen
- Aus der Abfolge der Positionen Bewegungsbahn, Richtung und Geschwindigkeit berechnen
- Messrauschen und kurzzeitige Aussetzer der Erkennung glätten bzw. überbrücken

### 4. Gestenerkennung
- Eine schnelle Wischbewegung als „Schnitt“ erkennen
- Langsame oder ruhende Bewegungen nicht als Schnitt werten

### 5. Spiel
- Eine einfache Fruit-Ninja-Variante umsetzen: Früchte fliegen ins Bild, der Spieler zerschneidet sie mit dem Stift
- Kamerakoordinaten auf das Spielfeld übertragen
- Prüfen, ob die Schnittbahn eine Frucht trifft
- Punkte zählen, optional Bomben und Leben ergänzen

## Anforderungen

- **Echtzeit:** Die Verzögerung zwischen Stiftbewegung und Reaktion im Spiel muss so gering sein, dass sich das Spiel direkt steuern lässt.
- **Robustheit:** Die Erkennung soll unter normalen Raumbedingungen ohne aufwendige Kalibrierung funktionieren.
- **Visualisierung:** Die Schnittspur des Stifts wird im Spiel angezeigt. Für die Entwicklung soll sich zusätzlich ein Debug-Fenster mit Kamerabild und erkannter Stiftposition einblenden lassen.

## Ergebnis

Am Ende steht ein lauffähiges Programm, mit dem man Fruit Ninja allein durch Stiftbewegungen vor einer USB-Kamera spielen kann.
