"""Fruit Ninja mit der Kamera: Ein farbiger Stift (oder Gegenstand) ist die Klinge.

Beim Start wird der Gegenstand einmal kalibriert: Aus den Farben im
Kalibrierfeld und denen des restlichen Bilds entsteht ein Farbmodell
(H-S-Histogramm), das angibt, wie wahrscheinlich eine Farbe zum Gegenstand
gehoert. Per Histogramm-Rueckprojektion wird der Gegenstand dann in jedem
Frame gesucht. Eine Frucht ist zerschnitten, wenn die Strecke, die der
Gegenstand seit dem letzten Frame zurueckgelegt hat, durch die Frucht geht
und er dabei schnell genug ist. Alles andere wird ignoriert.

Tasten:  Leertaste = Kalibrieren / Start,  r / k = neu kalibrieren,
         m = Farbmaske ein/aus,  q / ESC = beenden
Aufruf:  python fruit_ninja.py [kamera_index]
"""
import math
import os
import random
import sys
import time

import cv2
import numpy as np

from camera import CAMERA_INDEX, open_camera

W, H = 1280, 720
ROUND_TIME = 30          # Sekunden pro Runde
COUNTDOWN = 3            # Sekunden Countdown vor der Runde
GRAVITY = 1100           # px/s^2, hoeher = Fruechte fliegen schneller durchs Bild

# Farberkennung des Gegenstands, wird beim Start kalibriert (OpenCV-HSV: H 0-179, S/V 0-255)
DETECT_SCALE = 2                 # Farbsuche auf halber Aufloesung (640x360)
CALIB_BOX = 110                  # Kantenlaenge des Kalibrierfelds in px
CALIB_FRAMES = 15                # ueber so viele Frames wird das Farbmodell gemittelt
HIST_BINS = [30, 32]             # Histogramm ueber Farbton (H) und Saettigung (S)
HIST_RANGES = [0, 180, 0, 256]
MIN_SAT, MIN_VAL = 50, 40        # darunter ist der Farbton eines Pixels nicht verlaesslich
PROB_THRESHOLD = 150             # 0-255: ab hier gilt ein Pixel als Gegenstand
MIN_PEN_AREA = 40                # Mindestflaeche in Pixeln (auf halber Aufloesung)
MAX_JUMP = 400                   # groessere Spruenge pro Frame gelten als Fehlerkennung
MIN_SLICE_SPEED = 250            # px/s, langsamer bewegt schneidet der Stift nicht
TRAIL_TIME = 0.25                # Sekunden, die die Klingenspur sichtbar bleibt

FRUIT_SCALE = 0.8        # Fruechte im Spiel kleiner als die Grundgroesse
SPAWN_INTERVAL = (0.25, 1.8)     # zufaellige Pause zwischen zwei Wuerfen in Sekunden
ROUND_INTENSITY = (0.6, 1.4)     # pro Runde zufaellig: >1 = mehr Fruechte in dieser Runde
BOMB_CHANCE = (0.12, 0.3)        # Bomben-Wahrscheinlichkeit pro Wurf, Rundenstart -> -ende
BOMB_LOCK = 3.0                  # Sekunden, die nach einem Bombentreffer nichts schneidet

HIGHSCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "highscore.txt")
WINDOW = "Fruit Ninja"

# (Name, Schale BGR, Fruchtfleisch BGR, Radius)
FRUITS = [
    ("Apfel", (40, 40, 200), (180, 230, 245), 45),
    ("Orange", (0, 140, 255), (80, 190, 255), 45),
    ("Zitrone", (40, 230, 240), (170, 250, 250), 40),
    ("Melone", (40, 140, 30), (80, 70, 230), 65),
    ("Pflaume", (120, 30, 110), (120, 200, 230), 38),
    ("Kiwi", (40, 90, 120), (60, 200, 120), 40),
]
MELON = FRUITS[3]
BOMB = ("Bombe", (35, 35, 35), (0, 140, 255), 42)


def shade(color, f):
    return tuple(int(min(255, c * f)) for c in color)


class Fruit:
    def __init__(self, kind, x, y, vx, vy):
        self.name, self.skin, self.flesh, self.r = kind
        self.kind = kind
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.angle = random.uniform(0, 360)
        self.spin = random.uniform(-180, 180)

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt

    def gone(self):
        return self.vy > 0 and self.y - self.r > H

    def draw(self, img, r=None):
        r = r or self.r
        c = (int(self.x), int(self.y))
        cv2.circle(img, c, r, self.skin, -1, cv2.LINE_AA)
        if self.name == "Melone":
            for f in (0.35, 0.75):
                cv2.ellipse(img, c, (int(r * f), r), self.angle, 0, 360,
                            shade(self.skin, 0.5), 4, cv2.LINE_AA)
        cv2.circle(img, c, r, shade(self.skin, 0.6), 2, cv2.LINE_AA)
        # Glanzpunkt
        cv2.circle(img, (int(self.x - r * 0.35), int(self.y - r * 0.35)), max(2, r // 5),
                   shade(self.skin, 1.6), -1, cv2.LINE_AA)
        # Stiel und Blatt (drehen sich mit der Frucht)
        if self.name not in ("Melone", "Kiwi"):
            a = math.radians(self.angle)
            dx, dy = math.sin(a), -math.cos(a)
            top = (int(self.x + dx * r * 0.9), int(self.y + dy * r * 0.9))
            tip = (int(self.x + dx * r * 1.25), int(self.y + dy * r * 1.25))
            cv2.line(img, top, tip, (30, 60, 90), 4, cv2.LINE_AA)
            leaf = (int(tip[0] + dy * r * -0.25), int(tip[1] + dx * r * 0.25))
            cv2.ellipse(img, leaf, (max(3, r // 4), max(2, r // 9)), self.angle, 0, 360,
                        (40, 170, 40), -1, cv2.LINE_AA)


class Bomb(Fruit):
    def draw(self, img, r=None):
        r = r or self.r
        c = (int(self.x), int(self.y))
        cv2.circle(img, c, r, self.skin, -1, cv2.LINE_AA)
        cv2.circle(img, c, r, (0, 0, 220), 3, cv2.LINE_AA)
        cv2.circle(img, (int(self.x - r * 0.35), int(self.y - r * 0.35)), max(2, r // 5),
                   (110, 110, 110), -1, cv2.LINE_AA)
        # Zuendschnur mit flackerndem Funken
        a = math.radians(self.angle)
        dx, dy = math.sin(a), -math.cos(a)
        top = (int(self.x + dx * r * 0.9), int(self.y + dy * r * 0.9))
        tip = (int(self.x + dx * r * 1.4), int(self.y + dy * r * 1.4))
        cv2.line(img, top, tip, (90, 140, 170), 4, cv2.LINE_AA)
        spark = random.choice([(0, 200, 255), (0, 255, 255), (255, 255, 255)])
        cv2.circle(img, tip, random.randint(5, 9), spark, -1, cv2.LINE_AA)


class Popup:
    """Kurz eingeblendeter Text, z.B. der Punktabzug einer Bombe."""

    def __init__(self, text, x, y, color):
        self.text, self.x, self.y, self.color = text, x, y, color
        self.life = 1.0

    def update(self, dt):
        self.y -= 60 * dt
        self.life -= dt

    def gone(self):
        return self.life <= 0

    def draw(self, img):
        draw_text(img, self.text, (self.x, self.y), 2.0, self.color, 4, center=True)


class Half:
    """Eine Haelfte einer zerschnittenen Frucht."""

    def __init__(self, fruit, cut_angle, side):
        self.skin, self.flesh, self.r = fruit.skin, fruit.flesh, fruit.r
        self.x, self.y = fruit.x, fruit.y
        self.angle = cut_angle + (180 if side else 0)
        # Haelften fliegen senkrecht zur Schnittlinie auseinander
        push = math.radians(self.angle + 90)
        self.vx = fruit.vx + math.cos(push) * 220
        self.vy = min(fruit.vy, 0) + math.sin(push) * 220
        self.spin = random.uniform(-200, 200)

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt

    def gone(self):
        return self.y - self.r > H

    def draw(self, img):
        c = (int(self.x), int(self.y))
        cv2.ellipse(img, c, (self.r, self.r), self.angle, 0, 180, self.skin, -1, cv2.LINE_AA)
        inner = int(self.r * 0.85)
        cv2.ellipse(img, c, (inner, inner), self.angle, 0, 180, self.flesh, -1, cv2.LINE_AA)


class Particle:
    def __init__(self, x, y, color):
        a = random.uniform(0, 2 * math.pi)
        v = random.uniform(100, 450)
        self.x, self.y = x, y
        self.vx, self.vy = math.cos(a) * v, math.sin(a) * v
        self.color = color
        self.life = self.max_life = random.uniform(0.3, 0.7)

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

    def gone(self):
        return self.life <= 0

    def draw(self, img):
        r = max(1, int(6 * self.life / self.max_life))
        cv2.circle(img, (int(self.x), int(self.y)), r, self.color, -1, cv2.LINE_AA)


def slice_fruit(fruit, effects):
    cut = random.uniform(0, 180)
    effects.append(Half(fruit, cut, False))
    effects.append(Half(fruit, cut, True))
    for _ in range(18):
        effects.append(Particle(fruit.x, fruit.y, fruit.flesh))


def explode_bomb(bomb, effects):
    for _ in range(40):
        effects.append(Particle(bomb.x, bomb.y, random.choice(
            [(0, 140, 255), (0, 220, 255), (60, 60, 60), (200, 200, 200)])))
    effects.append(Popup("Gesperrt!", bomb.x, bomb.y, (60, 60, 255)))


def spawn_fruit(bomb=False):
    kind = BOMB if bomb else random.choice(FRUITS)
    x = random.uniform(W * 0.15, W * 0.85)
    y = H + kind[3]
    peak = random.uniform(H * 0.08, H * 0.45)
    vy = -math.sqrt(2 * GRAVITY * (y - peak))
    vx = (W / 2 - x) * random.uniform(0.3, 0.8) + random.uniform(-80, 80)
    fruit = (Bomb if bomb else Fruit)(kind, x, y, vx, vy)
    fruit.r = int(fruit.r * FRUIT_SCALE)
    return fruit


def to_hsv(frame):
    """Verkleinertes HSV-Bild und Maske der Pixel mit verlaesslichem Farbton."""
    small = cv2.resize(frame, (W // DETECT_SCALE, H // DETECT_SCALE))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    valid = cv2.inRange(hsv, (0, MIN_SAT, MIN_VAL), (180, 255, 255))
    return hsv, valid


def calib_box():
    """Kalibrierfeld (x0, y0, x1, y1) in Spielkoordinaten."""
    h = CALIB_BOX // 2
    return W // 2 - h, H // 2 - h, W // 2 + h, H // 2 + h


def calib_histograms(hsv, valid):
    """H-S-Histogramme vom Kalibrierfeld (Gegenstand) und vom restlichen Bild (Hintergrund).

    Liefert ausserdem den Anteil farbiger Pixel im Feld.
    """
    x0, y0, x1, y1 = (v // DETECT_SCALE for v in calib_box())
    obj_mask = np.zeros_like(valid)
    obj_mask[y0:y1, x0:x1] = valid[y0:y1, x0:x1]
    # Rand ums Feld nicht als Hintergrund werten: dort ragt meist noch der Gegenstand hinaus
    m = CALIB_BOX // DETECT_SCALE // 2
    bg_mask = valid.copy()
    bg_mask[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m] = 0
    obj = cv2.calcHist([hsv], [0, 1], obj_mask, HIST_BINS, HIST_RANGES)
    bg = cv2.calcHist([hsv], [0, 1], bg_mask, HIST_BINS, HIST_RANGES)
    colored = cv2.countNonZero(obj_mask) / ((x1 - x0) * (y1 - y0))
    return obj, bg, colored


def build_model(obj, bg, colored):
    """Farbmodell aus den aufsummierten Histogrammen.

    Jeder H-S-Bin bekommt die Wahrscheinlichkeit (0-255), dass ein Pixel dieser
    Farbe zum Gegenstand gehoert: Farben, die im Feld haeufig und im Hintergrund
    selten sind, zaehlen stark. Liefert (Modell oder None, Hinweistext).
    """
    if colored < 0.3:
        return None, "Zu wenig Farbe im Feld - nimm einen farbigen Gegenstand"
    # leicht verwischen, damit kleine Licht- und Farbschwankungen noch passen
    obj = cv2.GaussianBlur(obj, (3, 3), 0)
    bg = cv2.GaussianBlur(bg, (3, 3), 0)
    obj /= obj.sum()
    bg /= max(bg.sum(), 1.0)
    obj[obj < 0.005] = 0   # vereinzelte Farben im Feld sind Rauschen
    prob = obj / (obj + bg + 1e-9)
    # Wie viel des Gegenstands sich klar vom Hintergrund abhebt
    separable = obj[prob > PROB_THRESHOLD / 255].sum()
    message = ""
    if separable < 0.5:
        message = "Achtung: Farbe kommt auch im Hintergrund vor - Erkennung evtl. unsicher"
    return (prob * 255).astype(np.float32), message


def find_pen(hsv, valid, model):
    """Liefert die Position des kalibrierten Gegenstands (oder None) und seine Maske."""
    prob = cv2.calcBackProject([hsv], [0, 1], model, HIST_RANGES, 1)
    prob = cv2.bitwise_and(prob, valid)
    prob = cv2.GaussianBlur(prob, (5, 5), 0)
    _, mask = cv2.threshold(prob, PROB_THRESHOLD, 255, cv2.THRESH_BINARY)
    # kleine Stoerpixel entfernen, Luecken im Gegenstand schliessen
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    blob = max(contours, key=cv2.contourArea)
    if cv2.contourArea(blob) < MIN_PEN_AREA:
        return None, mask
    m = cv2.moments(blob)
    pos = (m["m10"] / m["m00"] * DETECT_SCALE, m["m01"] / m["m00"] * DETECT_SCALE)
    return pos, mask


def segment_hits(p, q, x, y, r):
    """True, wenn die Strecke p-q den Kreis um (x, y) mit Radius r schneidet."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else ((x - p[0]) * dx + (y - p[1]) * dy) / length2
    t = max(0.0, min(1.0, t))
    cx, cy = p[0] + t * dx, p[1] + t * dy
    return (cx - x) ** 2 + (cy - y) ** 2 <= r * r


def draw_blade(img, trail, now, locked):
    color = (150, 150, 150) if locked else (0, 165, 255)
    for (p, tp), (q, tq) in zip(trail, trail[1:]):
        f = 1.0 - (now - tq) / TRAIL_TIME
        if f <= 0:
            continue
        a, b = (int(p[0]), int(p[1])), (int(q[0]), int(q[1]))
        cv2.line(img, a, b, color, max(2, int(14 * f)), cv2.LINE_AA)
        cv2.line(img, a, b, (255, 255, 255), max(1, int(5 * f)), cv2.LINE_AA)
    if trail:
        p = trail[-1][0]
        cv2.circle(img, (int(p[0]), int(p[1])), 10, color, 2, cv2.LINE_AA)


def draw_text(img, s, pos, scale=1.0, color=(255, 255, 255), thick=2, center=False):
    font = cv2.FONT_HERSHEY_DUPLEX
    if center:
        (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
        pos = (pos[0] - tw // 2, pos[1] + th // 2)
    pos = (int(pos[0]), int(pos[1]))
    cv2.putText(img, s, pos, font, scale, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, s, pos, font, scale, color, thick, cv2.LINE_AA)


def load_highscore():
    try:
        with open(HIGHSCORE_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def save_highscore(value):
    with open(HIGHSCORE_FILE, "w") as f:
        f.write(str(value))


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_INDEX
    # in 1080p aufnehmen und selbst verkleinern: bei 720p liefert die ELP nur 9 fps
    cap = open_camera(index)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, W, H)

    highscore = load_highscore()
    # calibrate -> verify -> menu -> countdown -> play -> menu
    state = "calibrate"
    state_since = time.time()
    model = None              # Farbmodell des Gegenstands, entsteht bei der Kalibrierung
    calib_msg = ""
    calib_count = 0           # >0: Kalibrierung laeuft, so viele Frames gesammelt
    calib_obj = calib_bg = None
    calib_colored = 0.0
    last_score = None
    new_record = False
    score = 0
    fruits, effects = [], []
    next_spawn = 0.0
    intensity = 1.0
    lock_until = 0.0
    flash_until = 0.0
    show_mask = False
    trail = []                # [((x, y), zeit), ...] der letzten Stiftpositionen
    start_fruit = Fruit(MELON, W / 2, H / 2, 0, 0)
    start_fruit.spin = 0
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Kein Bild von der Kamera erhalten")
            break
        frame = cv2.flip(frame, 1)  # spiegeln, damit es sich wie ein Spiegel anfuehlt
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))

        now = time.time()
        dt = min(now - last, 0.05)
        last = now

        # Klinge = Strecke von der letzten zur aktuellen Stiftposition
        hsv, valid = to_hsv(frame)
        pen, mask = find_pen(hsv, valid, model) if model is not None else (None, None)
        blade = None
        if pen is None:
            trail.clear()
        else:
            if trail and math.dist(trail[-1][0], pen) > MAX_JUMP:
                trail.clear()
            if trail:
                prev_pen, prev_t = trail[-1]
                speed = math.dist(prev_pen, pen) / max(now - prev_t, 1e-6)
                if speed >= MIN_SLICE_SPEED:
                    blade = (prev_pen, pen)
            trail.append((pen, now))
        trail = [(p, t) for p, t in trail if now - t <= TRAIL_TIME]
        locked = now < lock_until
        if locked:
            blade = None

        elapsed = now - state_since
        key = cv2.waitKey(1) & 0xFF
        if key == ord("m"):
            show_mask = not show_mask

        if state == "calibrate":
            if calib_count == 0 and key == ord(" "):
                calib_count, calib_msg = 1, ""
                calib_obj = calib_bg = None
                calib_colored = 0.0
            if calib_count > 0:
                obj, bg, colored = calib_histograms(hsv, valid)
                calib_obj = obj if calib_obj is None else calib_obj + obj
                calib_bg = bg if calib_bg is None else calib_bg + bg
                calib_colored += colored / CALIB_FRAMES
                calib_count += 1
                if calib_count > CALIB_FRAMES:
                    calib_count = 0
                    model, calib_msg = build_model(calib_obj, calib_bg, calib_colored)
                    if model is not None:
                        state, state_since = "verify", now
        elif state == "verify":
            if key == ord(" "):
                state, state_since = "menu", now
            elif key == ord("r"):
                state, model, calib_msg = "calibrate", None, ""
        elif state == "menu":
            start_fruit.x = W / 2
            start_fruit.y = H / 2 + 15 * math.sin(now * 2)
            # kurze Sperrzeit, damit die Bewegung vom Bestaetigen nicht sofort startet
            hit = (elapsed > 1.0 and blade is not None
                   and segment_hits(*blade, start_fruit.x, start_fruit.y, 90))
            if key == ord("k"):
                state, model, calib_msg = "calibrate", None, ""
            elif hit or key == ord(" "):
                start_fruit.r = 90
                slice_fruit(start_fruit, effects)
                start_fruit.r = MELON[3]
                state, state_since = "countdown", now
        elif state == "countdown":
            if elapsed >= COUNTDOWN:
                state, state_since = "play", now
                score = 0
                lock_until = 0.0
                # jede Runde ist unterschiedlich dicht, damit nicht immer gleich viele kommen
                intensity = random.uniform(*ROUND_INTENSITY)
                next_spawn = now + random.uniform(0.2, 0.8)
        elif state == "play":
            remaining = ROUND_TIME - elapsed
            if now >= next_spawn:
                progress = elapsed / ROUND_TIME
                if random.random() < 0.1 * intensity:
                    count = random.randint(4, 6)    # gelegentliche Salve
                else:
                    count = random.choices([1, 2, 3], [5, 3, 1])[0]
                fruits.extend(spawn_fruit() for _ in range(count))
                bomb_chance = BOMB_CHANCE[0] + (BOMB_CHANCE[1] - BOMB_CHANCE[0]) * progress
                if random.random() < bomb_chance:
                    fruits.append(spawn_fruit(bomb=True))
                pause = random.uniform(*SPAWN_INTERVAL) / intensity
                next_spawn = now + pause * (1.0 - 0.3 * progress)
            for fruit in fruits[:]:
                # erst schneidbar, wenn die Frucht ganz im Bild ist
                if blade is None or fruit.y + fruit.r >= H:
                    continue
                if segment_hits(*blade, fruit.x, fruit.y, fruit.r):
                    fruits.remove(fruit)
                    if isinstance(fruit, Bomb):
                        explode_bomb(fruit, effects)
                        lock_until = now + BOMB_LOCK
                        flash_until = now + 0.25
                        blade = None    # Rest dieses Schnitts zaehlt auch nicht mehr
                    else:
                        slice_fruit(fruit, effects)
                        score += 1
            if remaining <= 0:
                last_score = score
                new_record = score > highscore
                if new_record:
                    highscore = score
                    save_highscore(highscore)
                state, state_since = "menu", now
                lock_until = 0.0

        for obj in fruits + effects:
            obj.update(dt)
        fruits = [f for f in fruits if not f.gone()]
        effects = [e for e in effects if not e.gone()]

        if now < flash_until:
            red = np.zeros_like(frame)
            red[:] = (0, 0, 255)
            frame = cv2.addWeighted(frame, 0.6, red, 0.4, 0)

        for obj in fruits + effects:
            obj.draw(frame)
        draw_blade(frame, trail, now, now < lock_until)

        if state == "calibrate":
            x0, y0, x1, y1 = calib_box()
            color = (0, 255, 0) if calib_count > 0 else (255, 255, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
            draw_text(frame, "Kalibrierung", (W // 2, 70), 1.5, (0, 255, 255), 3, center=True)
            draw_text(frame, "Halte deinen Stift / Gegenstand in das Feld,", (W // 2, 130), 0.9,
                      center=True)
            draw_text(frame, "so dass er es moeglichst ganz ausfuellt.", (W // 2, 170), 0.9,
                      center=True)
            if calib_count > 0:
                draw_text(frame, "Stillhalten...", (W // 2, y1 + 60), 1.2, (0, 255, 0), 2,
                          center=True)
            else:
                draw_text(frame, "Leertaste = Kalibrieren", (W // 2, y1 + 60), 1.2, center=True)
            if calib_msg:
                draw_text(frame, calib_msg, (W // 2, H - 60), 0.9, (60, 60, 255), 2, center=True)
        elif state == "verify":
            draw_text(frame, "Wird dein Gegenstand erkannt und verfolgt?", (W // 2, 70), 1.2,
                      center=True)
            draw_text(frame, "Bewege ihn durchs Bild - die Spur folgt ihm.", (W // 2, 120), 0.9,
                      center=True)
            draw_text(frame, "Leertaste = Weiter     r = Neu kalibrieren", (W // 2, H - 60), 1.1,
                      (0, 255, 255), 2, center=True)
            if calib_msg:
                draw_text(frame, calib_msg, (W // 2, 170), 0.8, (0, 200, 255), 2, center=True)
        elif state == "menu":
            start_fruit.draw(frame, 90)
            draw_text(frame, "Zerschneide die Melone mit deinem Gegenstand", (W // 2, 110), 1.2,
                      center=True)
            draw_text(frame, "(oder Leertaste,  k = neu kalibrieren)", (W // 2, 160), 0.8,
                      center=True)
            draw_text(frame, f"Bombe getroffen = {BOMB_LOCK:.0f} s gesperrt", (W // 2, 205), 0.9,
                      (60, 60, 255), 2, center=True)
            if last_score is not None:
                draw_text(frame, f"Punkte: {last_score}", (W // 2, H - 150), 1.6,
                          (0, 255, 255), 3, center=True)
                if new_record:
                    draw_text(frame, "Neuer Highscore!", (W // 2, H - 95), 1.2,
                              (0, 200, 255), 2, center=True)
            draw_text(frame, f"Highscore: {highscore}", (W // 2, H - 40), 1.0, center=True)
        elif state == "countdown":
            n = COUNTDOWN - int(elapsed)
            draw_text(frame, str(n), (W // 2, H // 2), 6, (0, 255, 255), 10, center=True)
        elif state == "play":
            remaining = max(0.0, ROUND_TIME - elapsed)
            draw_text(frame, f"Punkte: {score}", (30, 60), 1.4, (0, 255, 255), 3)
            draw_text(frame, f"Highscore: {highscore}", (30, 105), 0.9)
            time_color = (60, 60, 255) if remaining <= 5 else (255, 255, 255)
            draw_text(frame, f"Zeit: {math.ceil(remaining)}", (W - 230, 60), 1.4, time_color, 3)
            if now < lock_until:
                draw_text(frame, f"GESPERRT {lock_until - now:.1f} s", (W // 2, 60), 1.4,
                          (60, 60, 255), 3, center=True)

        if model is not None and pen is None:
            draw_text(frame, "Gegenstand nicht erkannt", (30, H - 30), 0.8, (150, 150, 150))
        if mask is not None and (show_mask or state == "verify"):
            # Maske des erkannten Gegenstands unten rechts, zur Kontrolle der Kalibrierung
            thumb = cv2.cvtColor(cv2.resize(mask, (320, 180)), cv2.COLOR_GRAY2BGR)
            frame[H - 190:H - 10, W - 330:W - 10] = thumb
            cv2.rectangle(frame, (W - 330, H - 190), (W - 10, H - 10), (0, 165, 255), 2)

        cv2.imshow(WINDOW, frame)
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
