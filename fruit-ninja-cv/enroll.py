"""Enroll a player for face identification.

Records a few aligned face crops from the webcam into ``faces/<Name>/``. Look
into the camera and turn your head slightly while it records.

Usage:  python enroll.py "Noah Braun" [--count 8]
Keys:   q = cancel
"""

import argparse
import logging
import os
import time

import cv2

import config
import face_id
from logging_config import setup_logging
from main import open_camera
from utils import draw_text

logger = logging.getLogger(__name__)

CAPTURE_INTERVAL = 0.5           # seconds between two saved crops


def main():
    parser = argparse.ArgumentParser(description="Spieler fuer die Gesichtserkennung einlernen")
    parser.add_argument("name", help='Name des Spielers, z. B. "Noah Braun"')
    parser.add_argument("--count", type=int, default=8, help="Anzahl Referenzbilder")
    args = parser.parse_args()

    setup_logging()
    folder = face_id.person_dir(os.path.join(face_id.BASE_DIR, config.FACES_DIR), args.name)
    os.makedirs(folder, exist_ok=True)
    paths = face_id.ensure_models()
    identifier = face_id.FaceIdentifier(paths["detector"], paths["recognizer"])
    logger.info("Enrolling %s into %s", args.name, folder)

    cap, _ = open_camera()
    saved, last_save = 0, 0.0
    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            faces = identifier.detect(frame)
            now = time.time()
            if len(faces) == 1 and now - last_save >= CAPTURE_INTERVAL:
                crop = identifier.aligned_crop(frame, faces[0])
                cv2.imwrite(os.path.join(folder, f"{int(now * 1000)}.png"), crop)
                saved, last_save = saved + 1, now
            for f in faces:
                x, y, w, h = (int(v) for v in f[:4])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 220, 80), 2)
            hint = "Genau ein Gesicht ins Bild" if len(faces) != 1 else "Kopf leicht drehen"
            draw_text(frame, f"{face_id.display_name(args.name)}: {saved}/{args.count}  -  {hint}",
                      (20, 40), scale=0.8)
            cv2.imshow("Spieler einlernen", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    logger.info("Saved %d reference image(s) for %s", saved, args.name)
    print(f"{saved} Referenzbilder fuer {args.name} in {folder} gespeichert.")


if __name__ == "__main__":
    main()
