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
| `lightsaber.py` | Lichtschwert fester Länge: Spitze am Stiftpunkt, Richtung zur erkannten Hand |
| `suit.py` | Animierter Ganzkörper-Raumanzug: Silhouette → Stoff, Helm mit Visier-Glanz, Brust-LEDs |
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
      **~5 ms/Frame** im Schnitt. Diese Maske speist Freistellung **und** den
      Ganzkörper-Anzug (`suit.py`). (Modell exportiert aus Ultralytics
      YOLOv8n-seg, AGPL-3.0, als Release-Asset gehostet.)
    - `off` – kein Compositing, rohes Kamerabild.
    Alle Modi nutzen dasselbe `composite_space`; nur die Maskenquelle unterscheidet
    sich (greenscreen und absdiff teilen zusätzlich die Nachbearbeitung `_refine_mask`).
    Die Wahl wird in `settings.json`
    (`background_mode`, via `cameras.load/save_background_mode`) **persistiert** –
    analog zur Kamerawahl. Default ist `normal` (unverändertes Verhalten).
- **Animierter Ganzkörper-Raumanzug (`suit.py`, Issues #27/#38).** Sobald der Spieler
  freigestellt ist, wird er zum **animierten Astronauten**. Frühere Ansätze (festes
  Anzug-Bild am Gesicht; nur der Torso wird Stoff, Kopf/Hände bleiben frei; optionales
  Textur-Bild `textures/suit.png`) sind **ersetzt**: ein Filmfiguren-Foto über den
  Körper zu strecken sah schlecht aus und deckte Löcher (Arm-Kreis) falsch ab, und der
  „nur Kleidung"-Ansatz ließ Kopf/Hände nackt. Entschieden – **der ganze Körper wird
  Anzug**, rein prozedural:
  - **Stoff:** Die **komplette Silhouette** (Rumpf, Arme, Hände = Handschuhe) wird zu
    kühlem Weiß, zum Rand hin dunkler (Distanztransformation → runde, schlauchartige
    Gliedmaßen), mit einem Hauch der echten Falten (Bild minus Weichzeichnung, Faktor
    0,3, max. ±30). Nur der **Stift** bleibt sichtbar (Maske des Marker-Trackers als
    `keep`). Keine Haut-/Kopf-Ausnahme mehr.
  - **Helm (animiert):** über dem Kopf (Gesichts-Box von YuNet, 320 px, ~3 ms; ohne
    Gesicht die Kopfregion der Bounding-Box) eine metallische Schale mit dunklem,
    bläulichem **Visier**. Über das Visier wandert ein **Glanz-Reflex** (Gauß-Streifen,
    dessen Position per `sin(phase)` läuft); oben blinkt ein Antennenlicht.
  - **Brustpanel (animiert):** Steuerpanel mit **blinkenden LEDs** (Helligkeit per
    `sin(phase + i)`), am Gesicht/Torso verankert, Position geglättet (`RectSmoother`).
  - **Animation** läuft über `phase = time.monotonic()`, das `dress()` übergeben wird.
  - **Masken-Quelle:** `dress()` nimmt die Spielermaske eines beliebigen Hintergrund-
    Modus (greenscheen/normal/**yolo**) entgegen; mit der präzisen YOLO-Maske sitzt der
    Anzug am saubersten (ohne Greenscreen). Rein klassische Bildverarbeitung (Masken,
    Distanztransformation, Zeichnen), keine Posenerkennung (die kostete ~100–200 ms).
  - **Performance:** Helligkeitskarte auf halber Auflösung und nur im Bildausschnitt
    des Spielers, Schattierung per Tabelle, Visier/Panel auf kleinen ROIs → wenige
    ms/Frame zusätzlich zum Compositing.
- **Stift als Lichtschwert, Griff auf der Hand (`lightsaber.py`).** Am Stift wird auf
  jedem Bildschirm ein Lichtschwert gezeichnet. Die Klingenspitze ist der Punkt, der
  schneidet (die Stiftspitze); der **Griff sitzt auf der Hand**, die Klinge läuft von
  der Spitze zur Hand und kippt mit ihr. Verlauf der Entscheidungen:
  1. Erste Fassung: Richtung aus der Form der Farbmaske (`fitLine`), Griff über dem
     Stift → im Live-Test unruhig, verworfen.
  2. Statisch senkrecht nach unten → ruhig, aber unrealistisch.
  3. Klinge von der Spitze bis zur erkannten Hand → Länge und Richtung sprangen mit
     jeder Ungenauigkeit der Handerkennung, im Live-Test verworfen.
  4. Jetzt: **Spitze am Stift**; die Hand bestimmt die **Richtung**, und die
     Klingenlänge pendelt sich langsam (5 % pro Frame, nur in übernommenen Frames,
     begrenzt) so ein, dass der Griff auf der Hand sitzt – danach praktisch konstant,
     Zittern der Handerkennung mittelt sich heraus.
     Hand = die zur Spitze nächste ausreichend große Hautfläche (YCrCb,
     `lightsaber.skin_mask`) innerhalb `SABER_HAND_RADIUS`, Gesicht ausgeschlossen, davon nur
     der Teil in Handgröße direkt am Stift. Gegen Zappeln: Richtung stark geglättet;
     ein Sprung > 45° wird erst übernommen, wenn er 4 Frames anhält; Neigung höchstens
     75° aus der Senkrechten (Haut oberhalb der Spitze zählt nie); ohne Hand erst kurz
     halten, dann langsam in die Senkrechte zurückschwenken.
  Kosten: Handsuche ~5 ms (Abstände nur für Hautpixel statt für das ganze Fenster);
  die Gesichtserkennung wird pro Frame nur einmal berechnet und mit dem Raumanzug
  geteilt. Die Hauterkennung (YCrCb, halbe Auflösung) liegt in `lightsaber.py`, seit
  der Ganzkörper-Anzug keine Haut-Ausnahme mehr braucht; sie läuft auf dem rohen
  Kamerabild, findet die Hand also auch, wenn der Anzug sie als Handschuh zeigt. Dicker Strahl (weißer Kern + zwei additive Leucht-Schichten).
  Der Marker-Tracker liefert im Modus `point_mode="tip"` (`config.PEN_POINT`) das
  obere Stiftende; sein Standardmodus bleibt `centroid`. Die Schnitt-Logik selbst ist
  unverändert.

- **Code-Review #41 (CR-01 bis CR-10).** Jeder Befund wurde zuerst auf dem aktuellen
  Stand reproduziert, dann per Regressionstest festgehalten und behoben. Für die
  Übergänge zwischen Modulen gibt es jetzt `tests/loop_harness.py`: Er lässt die
  echte Hauptschleife `main._run_game_loop` mit einer Fake-Kamera und
  ausgeschalteten OpenCV-Fenstern headless laufen.
  - **CR-01 Klinge: Messung getrennt vom sichtbaren Schweif.** `Blade` führt zwei Puffer.
    `points` ist der sichtbare Schweif, der beim Trackingverlust ausläuft. `_live`
    enthält die letzten zwei echten Messungen. Kollision (`segment`,
    `current_point`) nutzt nur `_live`. `None` leert `_live` sofort, und die erste
    Messung danach beginnt einen neuen Schweif. So schneidet die Klinge weder
    an der alten Position weiter noch quer über die Lücke.
  - **CR-05/06 Wahrnehmung zuerst, einmal pro Frame.** Der Stift wird direkt nach
    dem Kopieren des Kamerabilds getrackt, und zwar einmal für alle Zustände. Seine
    Maske (`tracker.last_mask`) geht an den Anzug. `suit.dress` sichert die
    Stiftpixel vor dem Zeichnen und setzt sie nach Brustpanel und Helm wieder ein.
    Der Stift bleibt dadurch in jeder Render-Stufe sichtbar.
  - **CR-02 Kamera gilt erst mit Bild als offen.** `cameras._open` verlangt
    innerhalb von `FIRST_FRAME_TRIES` Versuchen ein Bild. Ein Wechsel gibt die alte
    Kamera nur frei, wenn sie existiert. Scheitert auch die Rückkehr zur alten
    Kamera, wird das als Fehler geloggt und `index=None` gesetzt. `read()` liefert
    dann `(False, None)` statt abzustürzen.
  - **CR-09/10 YOLO ist optional und an die Quelle gebunden.** Inferenzfehler
    (`cv2.error`, `ValueError`, `IndexError`) und I/O-Fehler beim Laden werden einmal
    geloggt. Danach setzt der Segmentierer `failed`, und die Schleife schaltet ihn
    ab, statt pro Frame Fehler zu werfen. Der Maskencache hängt an
    `source=(Kameraindex, Modus-Session)` und wird nach einem Kamera- oder
    Moduswechsel neu berechnet. Eine alte Maske landet so nie auf einem neuen Bild.
  - **CR-03 Gesichtseinlernen = eine zusammenhängende Session.** `FaceScanner` duldet
    kurze Aussetzer bis `FACE_SCAN_GAP` (0,5 s). Ist das Gesicht länger aus dem Kreis,
    werden die gesammelten Bilder und Stimmen verworfen. So werden nie Gesichter
    zweier Personen zu einem Profil gemischt.
  - **CR-04 Grüner Stift ist mit Greenscreen unvereinbar.** Die Regel steht in
    `pen_allowed` / `pen_for_mode`. Im Menü ist das grüne Farbfeld im
    Greenscreen-Modus durchgestrichen. Ein Klick darauf zeigt einen Hinweis statt
    die Farbe zu wählen. Wird auf Greenscreen umgeschaltet, während Grün aktiv
    ist, fällt der Stift auf Pink zurück, mit Hinweis.
  - **CR-07 Highscore: Fehler unterscheiden, nichts verlieren.** `_read` meldet
    `missing`, `corrupt` oder `unreadable`. Eine nicht lesbare Datei (z. B. gesperrt)
    wird nie überschrieben. Eine kaputte Datei wird als `.corrupt` gesichert. Geschrieben
    wird über eine `.tmp`-Datei mit `os.replace`, also atomar. `add_entry` liefert
    `saved`, und der Game-Over-Bildschirm meldet „Highscore konnte nicht gespeichert
    werden“.
  - **CR-08 Bildschirm unter Windows.** Messung am Demo-Laptop (Windows 11, 200 %
    Skalierung):

    | | Wert |
    |---|---|
    | Logische Auflösung | 1440×960 (3:2) |
    | Physische Auflösung | 2880×1920 |
    | Vorher erkannt | Fallback 1280×720 (`system_profiler` fehlt unter Windows) |
    | Vorher Canvas | 1152×720 (16:10), im Vollbild auf 3:2 gestreckt, also ca. 6 % horizontale Verzerrung |

    Die Mauszuordnung ist korrekt. Versuch: Ein 800×450-Fenster zeigt ein
    400×300-Bild. Ein Klick bei (600, 225) wird als (300, 150) gemeldet.
    OpenCV rechnet Mausklicks also selbst in Bildkoordinaten um, und Klick
    und Dwell treffen trotz Streckung.
    Lösung: `detect_screen_size` liest unter Windows `GetSystemMetrics`. Das liefert
    logische Pixel, also dieselben Einheiten wie das Fenster, ohne DPI-Awareness.
    `canvas_size` übernimmt außerhalb von macOS das Seitenverhältnis des Bildschirms,
    deshalb gibt es keine Verzerrung mehr.
    Die Höhe ist auf `MAX_CANVAS_HEIGHT = 720` begrenzt. Gemessen mit der echten
    Schleife (Greenscreen, Anzug, Lichtschwert):

    | Canvas | Zeit pro Frame (Median) |
    |---|---|
    | 1152×720 | 45 ms |
    | 1080×720 | 45 ms |
    | 1440×960 | 88 ms |

    Auf dem Demo-Laptop gilt also: Canvas 1080×720 (3:2), Vollbild unverzerrt.
    macOS behält 16:10 bei voller Bildschirmhöhe (unverändert).

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
- **Windows:** Sound (`afplay`) ist auf macOS ausgelegt. Kamera-Auswahl und
  Bildschirmgröße sind für Windows gelöst (#31, #41 CR-08).

## Bewusste Nicht-Ziele
- Kein ML für die Stifterkennung (klassische Bildverarbeitung). Ausnahme:
  Spielererkennung per Gesicht (siehe oben).
