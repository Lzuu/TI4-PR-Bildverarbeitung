"""Live-Bild der ELP-USBFHD01 USB-Kamera mit OpenCV.

Tasten:  q / ESC = beenden,  s = Screenshot speichern
Aufruf:  python camera.py [kamera_index]
"""
import sys
import time

import cv2

# Index 0/1 = interne Laptop-Kameras (Front/Rück), 2 = ELP "HD USB Camera"
CAMERA_INDEX = 2
WIDTH, HEIGHT, FPS = 1920, 1080, 30


def open_camera(index, width=WIDTH, height=HEIGHT, fps=FPS):
    # MSMF statt DirectShow: Unter DSHOW greift MJPG nicht und die ELP liefert
    # bei 1080p (YUY2) nur 5 fps, mit MSMF sind es volle 30 fps.
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {index} konnte nicht geoeffnet werden")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX
    cap = open_camera(index)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Kamera {index} geoeffnet: {w}x{h}")

    cv2.namedWindow("ELP", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ELP", 960, 540)

    t_last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kein Bild von der Kamera erhalten")
            break

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
        t_last = now

        display = frame.copy()
        cv2.putText(display, f"{w}x{h}  {fps:.1f} fps", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("ELP", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            name = time.strftime("snapshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(name, frame)
            print("Gespeichert:", name)
        if cv2.getWindowProperty("ELP", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
