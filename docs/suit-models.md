# Modellquellen

Es werden zur Laufzeit ausschließlich lokale ONNX-Dateien über `cv2.dnn` genutzt.
Modell-Dateien bleiben in `fruit-ninja-cv/models/` und sind durch .gitignore ausgeschlossen.

## Zusätzliche Pose-Modelle

Quelle und Lizenz: OpenCV Model Zoo, jeweils Apache-2.0 (vollständige Lizenz am Quelllink).

- [BlazePose Person Detector](https://github.com/opencv/opencv_zoo/tree/main/models/person_detection_mediapipe)
  – `person_detection_mediapipe_2023mar.onnx`, 11.990.159 Byte,
  SHA256 `47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f`.
- [BlazePose Landmarks](https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe)
  – `pose_estimation_mediapipe_2023mar.onnx`, 5.557.238 Byte,
  SHA256 `9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f`.

Vor-/Nachverarbeitung wurde auf Basis der dokumentierten Tensorlayouts und
Koordinatenkonventionen eigenständig für dieses Spiel implementiert. Die großen
statischen Ankertabellen des Beispiels werden als SSD-Gitter berechnet.

Keine Installation der Pakete `mediapipe`, `torch` oder `ultralytics` erforderlich.
Die bisherigen YuNet-/SFace-/YOLO-Quellen stehen weiterhin in face_id.py/yolo_seg.py.
Das YOLO-Modell und dessen bestehende Lizenzentscheidung wurden nicht ersetzt.
Die NASA-Bilder sind bereits freigegebene öffentliche Testdaten; siehe
fruit-ninja-cv/tests/data/QUELLEN.md. Es wurden keine privaten Kamerabilder gespeichert.
