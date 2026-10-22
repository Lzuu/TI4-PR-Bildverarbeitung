# Architektur

Architekturentscheidungen und Aufbau des Fruit-Ninja-CV-Spiels
(`fruit-ninja-cv/`).

## Überblick
Das Spiel ist in **kleine, fokussierte Module** aufgeteilt (eine Verantwortung je
Datei). `main.py` ist der Einstieg und hält die **Game-Loop** sowie die
**State-Machine** (Kalibrierung → Spiel → Game Over). Die Bildverarbeitung, die
Spiel-Physik und das Rendering sind davon getrennt.

```
Kamera ─► main (Loop, States, Input)
             │
             ├─► MarkerTracker   : Bild → HSV-Segmentierung → Stift-Position
             ├─► Blade           : letzte Positionen = Klinge (Trail, Segment)
             ├─► Game            : Spawning, Physik-Update, Kollision, Score/Leben
             │      └─► Fruit     : Projektil-Physik + prozedurale Zeichnung
             ├─► utils           : Geometrie (Linie-Kreis), Text-HUD
             └─► logging_config  : Logfiles (logs/)
```

## Module & Verantwortlichkeiten
| Modul | Verantwortung |
|-------|---------------|
| `main.py` | Game-Loop, State-Machine (FaceScan/NameEntry/Menu/Difficulty/Play/GameOver), Kamera-Capture, Dwell-Auswahl, Maus-Farbeinstellung, Rendering, Logging-Setup, Fehlerbehandlung |
| `config.py` | Gemeinsame Konstanten (Kamera, Marker, Dwell, Kombos, Highscore, HUD) |
| `colors.py` | Feste Stiftfarben Pink/Gelb/Grün (HSV-Bereich + UI-Farbe) |
| `difficulty.py` | Schwierigkeitsgrade (Tempo; immer 3 Leben) als Presets |
| `dwell.py` | VR-Dwell-Auswahl (Stift n Sekunden auf einem Ziel = Bestätigung) |
| `marker_tracker.py` | HSV-Farb-Segmentierung (eine oder mehrere Farben) + Positionsbestimmung |
| `blade.py` | Kurzer Trail der letzten Marker-Punkte; aktuelles Segment & Geschwindigkeit |
| `fruit.py` | Frucht-/Bomben-Entität: Projektil-Physik + prozedurale Darstellung |
| `game.py` | Spielzustand: Spawning (Bursts), Kollision, Punkte, Leben, Kombos |
| `heart.py` | Herz-Sprite (parametrische Herzkurve, Verlauf+Glanz) für die Leben |
| `background.py` | Spieler freistellen (Referenz-Differenz) und vor das Weltall setzen |
| `suit.py` | Raumanzug: Silhouette wird zu Anzugstoff (folgt den Armen), Kopf/Hände/Stift frei, Details am Gesicht |
| `cameras.py` | Kameras finden (Indizes durchprobieren), öffnen (1080p MJPG, unter Windows MSMF), wechseln, Auswahl in `settings.json` merken |
| `highscore.py` | Top-3-Highscore **je Schwierigkeit** mit Spielername (JSON-Persistenz) |
| `face_id.py` | Spielererkennung per Gesicht: YuNet-Detektion, SFace-Merkmale, Abgleich mit `faces/`, Scan im Kreis (`FaceScanner`), Einlernen |
| `keyboard.py` | Bildschirmtastatur (Dwell pro Taste) für den Namen neuer Spieler |
| `enroll.py` | Spieler einlernen: speichert ausgerichtete Gesichtsausschnitte nach `faces/<Name>/` |
| `sound.py` | Soundeffekte (synthetisiert, non-blocking via afplay) |
| `utils.py` | Wiederverwendbare Helfer: Linie-Kreis-Abstand, Text mit Schatten |
| `logging_config.py` | Zentrales Logging (rotierendes Logfile + Konsole) |

## Datenfluss (pro Frame)
1. `main` liest Frame, skaliert auf feste Größe, spiegelt (Mirror).
2. `MarkerTracker.track(frame)` → HSV → `inRange` → Morphologie → größte Kontur →
   Schwerpunkt = Stiftposition (oder `None`).
3. `Blade.add(point)` hält die letzten N Punkte (die Klinge).
4. `Game.update(blade)` wirft Früchte ein, bewegt sie (Schwerkraft), prüft
   **Kollision** (Abstand Frucht-Mittelpunkt ↔ Klingen-Segment ≤ Radius).
5. `Game.draw` + `Blade.draw` + HUD zeichnen ins Frame; `main` zeigt es an.

## Architekturentscheidungen (ADR-Kurzform)
- **Klassische Bildverarbeitung statt ML.** Farb-Segmentierung (HSV) + Konturen
  statt MediaPipe o. Ä. → passt zur Kursintention, transparente Pipeline, kein
  Modell-Overhead.
- **Feste Farbbereiche statt Selbst-Kalibrierung.** Pink/Gelb/Grün sind in
  `colors.py` als feste HSV-Bereiche hinterlegt (Hue als Hauptkriterium, S/V mit
  Mindestwert gegen Haut/heller Wand). Kein fehleranfälliges Live-Kalibrieren mehr.
  **Standard ist Pink, und es wird immer nur die EINE aktive Farbe getrackt**
  (keine Union) → keine Fehldetektion durch andere farbige Objekte.
- **Stiftfarbe als versteckte Einstellung (Maus).** Die Farbe wechselt man nur, wenn
  man will: Sie liegt hinter einem **„Stiftfarbe"-Button** im Menü; erst ein
  **Maus-Klick** öffnet die Farb-Kacheln. Das hält das Menü aufgeräumt und umgeht
  das Henne-Ei-Problem (einen andersfarbigen Stift könnte man nicht per Dwell
  wählen, solange er nicht getrackt wird) — Tracking bleibt einfarbig und robust.
- **Fullscreen & auflösungs-responsive.** Beim Start wird die Bildschirmgröße
  crash-frei über `system_profiler` ermittelt (nicht `tkinter` — das bricht mit
  OpenCVs Cocoa-Fenster ab). Gerendert wird auf ein festes **16:10**-Format
  (`config.ASPECT`), das der nutzbaren MacBook-Vollbildfläche entspricht → **kein
  grauer Rand**. Die 16:9-Webcam wird auf 16:10 **zugeschnitten** (nicht gestaucht)
  → keine Verzerrung. Ein globaler Faktor `config.SCALE = HEIGHT / REF_HEIGHT`
  skaliert UI (`utils.draw_text`, Kreise) **und Physik** (Radius, Wurf, Gravitation
  in `fruit.py`); Radius und Geschwindigkeit/Gravitation werden gemeinsam skaliert,
  sodass die Flugbahn proportional bleibt und die Airtime (in Frames) gleich.
- **Herz als Sprite statt Primitive.** Das Herz wird aus der parametrischen
  Herzkurve mit Farbverlauf, Glanz und Kontur in ein gecachtes BGRA-Sprite gerendert
  (`heart.py`) und per Alpha-Blending gezeichnet → sieht wie ein echtes Spiel-Asset
  aus, ohne externe Bilddatei.
- **VR-Dwell-Auswahl.** Start, Schwierigkeit und die Game-Over-Felder (Neustart/
  Startmenü/Quit) werden ausgewählt, indem der Stift einige Sekunden in einem Kreis
  gehalten wird (Ladering). Entkoppelt in `dwell.py` (zeitbasiert, framerate-
  unabhängig, mit injizierbarer Uhr → gut testbar).
- **Farbige Marker statt Hautfarbe.** Gesättigte Pen-Farben liegen weit vom Hautton
  entfernt → keine Verwechslung mit der Hand.
- **Projektil-Physik.** Früchte starten unter dem unteren Rand mit Aufwärts-
  Geschwindigkeit; Schwerkraft erzeugt die Parabel. Launch-Parameter sind so
  gewählt, dass der Scheitel **im Bild** bleibt (nicht oben abgeschnitten).
- **Kollision per Berührung.** Schneiden erfordert keine Mindestgeschwindigkeit;
  es zählt der Abstand zur Klinge (Punkt bei Ruhe, Segment bei Bewegung).
- **Prozedurale Grafik.** Früchte werden gezeichnet (Kreis + Highlight + Stiel) →
  keine Asset-Abhängigkeiten, läuft überall sofort.
- **State-Machine.** Klar getrennte Zustände (Kalibrierung/Play/GameOver) halten die
  Loop übersichtlich.
- **Zentrale Konfiguration.** Alle Stellschrauben in `config.py` → einfaches Tuning,
  keine Magic Numbers im Code.
- **Schwierigkeitsgrade als Presets.** `difficulty.py` bündelt die variierenden
  Werte je Stufe. Die Schwierigkeit steckt bewusst im **Tempo** der Früchte
  (höhere Gravitation + Wurfgeschwindigkeit + schnelleres Spawnen); **Leben bleiben
  immer 3**. `Game`/`Fruit` bekommen die gewählte Stufe injiziert → keine global
  verstreuten Magic Numbers. Jede Stufe hat ein **eigenes Leaderboard**.
- **Kombos über ein Zeitfenster.** Mehrere Treffer innerhalb `COMBO_WINDOW_FRAMES`
  zählen als ein Swipe; ab `COMBO_MIN` Früchten gibt es Bonuspunkte.
- **Spielererkennung per Gesicht mit vortrainierten Netzen (Issue #28).** Vor der
  Schwierigkeitsauswahl erkennt das Spiel den Spieler und ordnet ihm den Highscore
  zu. Klassische Verfahren (Haar-Cascade + LBPH) gibt es in OpenCV 5 nicht mehr
  (Haar nur noch in OpenCV 4, LBPH nur in opencv-contrib). Das Team hat sich deshalb
  für die in OpenCV 5 enthaltenen DNN-Verfahren entschieden: **YuNet** findet
  Gesichter, **SFace** berechnet aus dem ausgerichteten 112x112-Ausschnitt einen
  Merkmalsvektor; gleiche Person ab Kosinus-Ähnlichkeit 0,363. Genauer und
  lichtrobuster als LBPH, aber eine bewusste Ausnahme von „kein ML“ – nur für die
  Spielererkennung, nicht für die Stifterkennung. Die Modelle (OpenCV Model Zoo,
  ca. 39 MB) werden beim ersten Start nach `models/` geladen und per SHA256 geprüft,
  nicht ins Repo eingecheckt. Ohne Modelle (offline) startet das Spiel direkt im
  Menü und jeder spielt als Gast.
- **Gesichtsscan als erster Bildschirm, Einlernen im Spiel.** Statt eines separaten
  Einlern-Programms hält man das Gesicht in einen Kreis: Ein bekannter Spieler gilt
  nach 5 übereinstimmenden Frames als erkannt (nicht nach einem Einzelbild); bei
  einem unbekannten Gesicht werden 8 ausgerichtete Ausschnitte im Abstand von 0,3 s
  aufgenommen (leichte Kopfbewegungen → robustere Referenz), danach wird der Name
  abgefragt. Gescannt wird beim Start und über „Spieler wechseln“, nicht vor jeder
  Runde. Die Namenseingabe läuft per Stift über eine Bildschirmtastatur (Dwell
  0,8 s statt 3 s pro Taste, sonst zu langsam); die echte Tastatur geht zusätzlich.
  Logik (`FaceScanner`, `NameEntry`) ist von der Darstellung getrennt und mit
  injizierbarer Uhr testbar.
- **Referenzbilder sind biometrische Daten.** `faces/` liegt nur lokal (in
  `.gitignore`); die Tests nutzen gemeinfreie NASA-Fotos (`tests/data/QUELLEN.md`).
- **Kameraauswahl: Suche einmal beim Start, vor dem Öffnen (Issue #31).** OpenCV
  kennt keine Kameraliste, also werden die Indizes 0–4 durchprobiert (ca. 1 s je
  Kamera, auf dem Test-Laptop 2,1 s). Eine Suche im Hintergrund während des
  Spiels wurde verworfen: Front- und Rückkamera eines Laptops hängen oft am selben
  Controller, der nur einen Datenstrom liefert – die laufende Kamera lieferte
  während der Suche keine Bilder mehr. Beim Wechsel wird die alte Kamera vor dem
  Öffnen der neuen freigegeben; scheitert die neue, bleibt die alte. Die Wahl steht
  in `settings.json` (lokal, nicht in git); fehlt die Kamera beim nächsten Start,
  wird `CAMERA_INDEX` bzw. irgendeine gefundene genommen. Unter Windows wird MSMF
  mit MJPG verwendet (DirectShow: nur 5 fps bei 1080p). Der Kamera-Button steht
  auch auf dem Gesichtsscan, dem ersten Bildschirm – sonst käme man mit der
  falschen Kamera weder durch den Scan noch ins Menü.
- **Highscore als JSON.** `highscore.py` kapselt Laden/Schreiben der Top-3 und ist
  so unabhängig testbar (defekte/fehlende Datei → leere Liste).
- **Sound entkoppelt & non-blocking.** `Game` ruft nur `sounds.play(name)`; in Tests
  wird ein `NullSound` injiziert. WAVs werden synthetisiert (keine Asset-Dateien),
  Wiedergabe via `afplay` ohne die Game-Loop zu blockieren.
- **Logging statt Prints.** Rotierendes Logfile unter `logs/`; unbehandelte Fehler
  werden mit Traceback geloggt → nachvollziehbares Debugging nach einem Problem.
- **Test-case-basiert.** Die gesamte Logik (Geometrie, Tracking, Physik, Spiel) ist
  headless über `pytest` abgesichert und unabhängig von Kamera/GUI testbar.
- **Weltraum-Hintergrund per Referenz-Differenz (`background.py`).** Der Spieler wird
  freigestellt und vor einen Weltraum-Hintergrund gesetzt. Verfahren: einmalig ein
  **Referenzbild** des leeren Hintergrunds aufnehmen (Countdown beim Start, Taste `b`
  zum Neuaufnehmen), dann pro Frame `cv2.absdiff` → Graustufen → `threshold` →
  Morphologie (Open/Close) → Vordergrund-Maske; Hintergrund-Pixel werden durch das
  Weltall ersetzt (`composite_space`). Bewusst das **einfache, deterministische**
  Verfahren statt MOG2: reine Funktionen → headless testbar, keine adaptive Drift
  (ein stillstehender Spieler verschwindet nicht). Grenze: setzt eine **statische
  Kamera + gleichbleibendes Licht** voraus. Das **Tracking bleibt unberührt** – es
  arbeitet weiter auf dem rohen `clean`-Frame, nicht auf dem komponierten Bild. Das
  Weltall ist ein **statisches Asset** (`assets/space.png`), das – wie die Sounds –
  beim ersten Start generiert wird, falls es fehlt, und durch ein echtes Foto
  ersetzbar ist.
  - **Performance:** Gemessen ist der teuerste Schritt **nicht** die Morphologie
    (~1 ms full-res), sondern das **Compositing**: `np.where` über das ganze Frame
    kostet ~5,7 ms. Lösung: Compositing per **`cv2.copyTo`** (optimierte maskierte
    Kopie, in-place, ~0,7 ms); Segmentierung bleibt in **voller Auflösung**
    (Downscaling lohnt nicht – die Resizes kosten mehr, als die kleinere Morphologie
    spart). Gesamt-Pipeline **~2,5 ms/Frame** statt ~8,5 ms (~3,4×).
  - **Gesicht sauber freistellen (gefüllte Silhouette).** Wo der Hautton der Wand
    ähnelt, fällt die Differenz unter den Schwellwert → Löcher im Gesicht. Daher wird
    die Maske nach der Morphologie zu einer **gefüllten Silhouette** gemacht: externe
    Konturen oberhalb `config.BG_MIN_BLOB_FRAC` werden **ausgefüllt** gezeichnet →
    Innenlöcher verschwinden, Mini-Blobs (Rauschen) fallen raus.
  - **Referenz nur bewusst (Taste `b`), kein Auto-Capture.** Der Spieler muss zur
    Menübedienung im Bild stehen – ein automatisches Aufnehmen beim Start würde seine
    Silhouette ins Referenzbild „einbrennen". Daher wird die Referenz erst per `b`
    (mit Countdown zum Aus-dem-Bild-Gehen) aufgenommen; bis dahin zeigt das Spiel das
    rohe Kamerabild mit einem dezenten Hinweis.
  - **Vier Hintergrund-Modi (wählbar im Menü).** Über einen „Hintergrund"-Button
    (Maus-Klick, links unten über „Stiftfarbe") schaltet man zyklisch zwischen
    `greenscreen` / `normal` / `yolo` / `off` (`background.MODES`, `next_mode`):
    - `greenscreen` – **Chroma-Key**, robust gegen ungleichmäßiges Licht, damit nie
      Grün durchscheint (nicht nur die Ränder weggeschnitten): zwei billige C-Detektoren
      werden **ver-odert** – `cv2.inRange` über einen großzügigen HSV-Bereich
      (`config.GREEN_LOWER_HSV`/`GREEN_UPPER_HSV`) **und** eine helligkeitsunabhängige
      **Grün-Dominanz** `G − max(B,R) ≥ GREEN_DOMINANCE` (sättigende `cv2.subtract`),
      die auch dunkles/verwaschenes Grün außerhalb der HSV-Box erfasst. Der Backdrop
      wird per **`MORPH_CLOSE`** zuerst verdichtet (vereinzelte Nicht-Grün-Pixel im
      Grünfeld werden absorbiert → ganzes Feld keyt aus, nicht nur die Kante), dann
      `bitwise_not` → `MORPH_OPEN`. Die Silhouette wird **bewusst nicht gefüllt**
      (anders als beim absdiff-Verfahren): Grün *innerhalb* der Körperkontur – etwa
      im Kreis der Arme – ist echter Hintergrund und muss ausgekeyt bleiben. Rausch-
      Blobs entfernt stattdessen `_drop_small_blobs` per `connectedComponentsWithStats`
      + LUT-Relabeling, **ohne Löcher zu schließen**. Braucht **keine** Referenz-
      aufnahme und kommt mit bewegtem Spieler klar. Kosten **~5 ms/Frame** (volle
      Auflösung); Downscaling hilft kaum und macht die Kanten blockig → bewusst volle
      Auflösung (wie beim absdiff-Verfahren).
    - `normal` – das oben beschriebene **absdiff-Referenz**-Verfahren (Taste `b`).
    - `yolo` – **YOLOv8-seg Personensegmentierung** (`yolo_seg.py`, Issue #38): schneidet
      den Spieler aus **jedem** Hintergrund frei, **ohne Greenscreen und ohne Referenz**.
      Läuft als **ONNX über `cv2.dnn`** – wie die Gesichtserkennung (YuNet/SFace) und
      bewusst **ohne `torch`/`ultralytics`** zur Laufzeit; das Modell (`yolov8n-seg.onnx`,
      ~13 MB) wird beim ersten Start nach `models/` geladen (SHA256-geprüft), fehlt es,
      fällt der Modus sauber auf „aus" zurück. Der Decode der rohen YOLO-Ausgaben
      (Boxen + 32 Masken-Koeffizienten × 32 Prototyp-Masken, Person-Klasse, NMS) liegt in
      **reinen, headless getesteten** Funktionen. Inferenz (~15 ms) läuft nur jeden
      `YOLO_EVERY_N`-ten Frame auf 320px-Letterbox, die Maske wird dazwischen gehalten →
      **~5 ms/Frame** im Schnitt. Diese Maske speist Freistellung **und** Anzug; die
      `suit.py`-Pipeline bleibt unverändert. (Modell exportiert aus Ultralytics
      YOLOv8n-seg, AGPL-3.0, als Release-Asset gehostet.)
    - `off` – kein Compositing, rohes Kamerabild.
    Alle Modi nutzen dasselbe `composite_space`; nur die Maskenquelle unterscheidet
    sich (greenscreen und absdiff teilen zusätzlich die Nachbearbeitung `_refine_mask`).
    Die Wahl wird in `settings.json`
    (`background_mode`, via `cameras.load/save_background_mode`) **persistiert** –
    analog zur Kamerawahl. Default ist `normal` (unverändertes Verhalten).
- **Raumanzug aus der eigenen Silhouette (`suit.py`, Issue #27).** Sobald der Spieler
  freigestellt ist, trägt er einen Raumanzug, der sich **mit Armen und Körper
  bewegt**. Erster Ansatz war ein festes Anzug-Bild am Gesicht – das bewegt sich
  aber nicht mit den Armen. Eine Posenerkennung per neuronalem Netz könnte die
  Arme verfolgen, kostet auf dem Laptop aber ~100–200 ms pro Frame. Entschieden:
  Die **Vordergrund-Maske selbst wird zum Anzugstoff**. Anzug-Bereich = Silhouette
  ohne Kopf (Ellipse um das Gesicht + Haare darüber; ohne Gesicht die oberen 30 %
  der Bounding-Box), ohne Hände (Hautfarbe im YCrCb-Raum) und ohne Stift (Maske des
  Marker-Trackers). Der Stoff ist hellgrau/weiß, zum Silhouettenrand dunkler
  (Distanztransformation → Arme wirken rund) und behält die feinen Hell-Dunkel-
  Details der echten Kleidung (Bild minus Weichzeichnung → Falten bleiben). Kragen,
  Brustpanel und Abzeichen kommen als Sprite dazu, am **Gesicht** verankert
  (YuNet-Detektion ohne Wiedererkennung, 320 px, ~3 ms; Kragen unter dem Kinn,
  Breite 3,2 × Gesichtsbreite, Position geglättet). Das Sprite wird beim ersten
  Start nach `assets/suit_details.png` generiert (neuer Dateiname, damit ein altes,
  starres Anzug-Sprite nicht weiterverwendet wird). Grenze: Kleidung in Hauttönen
  bleibt sichtbar.
  - **Deckender Stoff, eigenes Anzugbild:** Die Falten der echten Kleidung fließen
    nur noch schwach und begrenzt ein (Faktor 0,3, max. ±30 Graustufen) – mit
    Faktor 1,2 schienen Muster des Shirts durch. Liegt `textures/suit.png` vor,
    wird dieses Bild über den Körper gestreckt – oben am **Kinn** verankert, nicht
    an der Oberkante der Silhouette (sonst rutscht z. B. der Gürtel auf die Brust,
    sobald die Arme gehoben werden); darüber liegt die Durchschnittsfarbe des
    Anzugs. Es folgt dem Körper, die Ärmel im Bild treffen aber nicht exakt die
    echten Arme. Mit eigenem Bild entfallen Kragen/Panel/Abzeichen. Das Bild selbst
    wird nicht eingecheckt (`.gitignore`), weil Anzugbilder oft urheberrechtlich
    geschützt sind.
  - **Performance:** Die erste Fassung kostete ~70 ms pro Frame (Weichzeichnen und
    Gleitkomma-Rechnung in voller Auflösung). Jetzt: Helligkeitskarte auf halber
    Auflösung und nur im Bildausschnitt des Spielers, Schattierung per Tabelle,
    Hauterkennung auf halber Auflösung, Überblenden per `cv2.blendLinear` →
    insgesamt ~9–19 ms pro Frame (je nach CPU-Takt).

## Qualitätssicherung
- **Tests:** `fruit-ninja-cv/tests/` (pytest), laufen ohne Kamera/GUI.
- **Logging:** `logs/fruit-ninja.log` (rotierend), Konsole zeigt nur Warnungen/Fehler.
- **CI:** GitHub Actions (`.github/workflows/tests.yml`) führt bei jedem Push und
  Pull Request alle Tests aus. In `main` kommt nur Code mit grünen Tests.

## Prozessentscheidungen
- **Noahs Programm ist die gemeinsame Basis.** Es gab zwei parallele Umsetzungen
  (Branches `FruitninjaLeon` und `FruitninjaNoah`). Das Team hat sich für Noahs
  Projekt entschieden (modularer Aufbau, mehr Funktionen, umfangreichere Tests).
  `FruitninjaNoah` wurde mit seiner Historie in `main` übernommen; Leons Stand bleibt
  im Branch `FruitninjaLeon` erhalten und kann als Ideenquelle dienen (z. B.
  Kalibrierung per Farbhistogramm, Schnitt erst ab Mindestgeschwindigkeit,
  Kamera-Setup für die ELP-USB-Kamera unter Windows).
- **Maßgeblich ist `aufgabenstellung.md`.** Noahs frühere `Aufgabenstellung.md`
  wurde nicht übernommen (veraltet; kollidiert unter Windows wegen gleicher
  Schreibweise bis auf Groß-/Kleinschreibung).
- **Iterationen über GitHub.** Jede Iteration ist ein Milestone, Aufgaben sind
  Issues, ein Branch pro Issue, Änderungen nur per Pull Request mit Tests.

## Offen gegenüber der Aufgabenstellung
- **Netzwerk-Mehrspieler (Teilaufgabe 6)** – fehlt noch, Architektur offen.
- **Schnittgeste (Teilaufgabe 4):** Aktuell schneidet schon Berührung ohne
  Mindestgeschwindigkeit; gefordert ist, dass langsame/ruhende Bewegungen kein
  Schnitt sind.
- **Tracking (Teilaufgabe 3):** Glättung von Messrauschen und Überbrücken kurzer
  Erkennungsaussetzer.
- **Stiftspitze (Teilaufgabe 2):** Aktuell wird der Schwerpunkt verwendet.
- **Debug-Fenster** mit Kamerabild und erkannter Stiftposition (bisher Maske per `d`).
- **Windows:** Standard-Kamera (`CAMERA_INDEX`) und Sound (`afplay`) sind auf macOS
  ausgelegt; unter Windows mit der ELP-USB-Kamera prüfen.

## Bewusste Nicht-Ziele
- Kein ML für die Stifterkennung (klassische Bildverarbeitung). Ausnahme:
  Spielererkennung per Gesicht (siehe oben).
