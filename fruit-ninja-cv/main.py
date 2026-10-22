"""Fruit Ninja CV -- entry point and game loop.

The window opens in fullscreen and the whole layout + physics scale to the
screen resolution (responsive). The app starts in a menu with the PINK pen
active by default (only pink is tracked). You start by holding the pen on the
"Start" circle (VR-style dwell). The pen colour is a *setting*: it is hidden
behind a "Stiftfarbe" button and only shown when you open it (mouse click).
With several cameras, a "Kamera" button lets you pick one the same way.

States:
  FACE_SCAN  First screen: face into the circle -> known player is greeted,
             a new face is recorded (only if face identification is available).
  NAME_ENTRY A new player types their name with the pen (on-screen keyboard).
  MENU       Start / Quit / Spieler wechseln (pen dwell); pen colour via mouse.
  DIFFICULTY Pick the difficulty (Einfach / Mittel / Schwer) by dwelling.
  PLAY       Slice thrown fruits with the pen (touch is enough), avoid bombs.
  GAME_OVER  Score + Top-3 leaderboard; dwell on Neustart / Startmenue / Quit.

Keys: q quit | d mask (debug) | b (re)capture the empty background for space mode
"""

import logging
import os
import re
import subprocess
import sys
import time

import cv2

import config
import background
import suit
import yolo_seg
from marker_tracker import MarkerTracker
from blade import Blade
from game import Game
from utils import draw_text
from logging_config import setup_logging
from difficulty import LEVELS
from colors import COLORS
from dwell import DwellTracker
from sound import SoundPlayer
import cameras
import face_id
import highscore
import keyboard

logger = logging.getLogger(__name__)

STATE_FACE_SCAN = "face_scan"
STATE_NAME_ENTRY = "name_entry"
STATE_MENU = "menu"
STATE_DIFFICULTY = "difficulty"
STATE_PLAY = "play"
STATE_GAME_OVER = "game_over"


def detect_screen_size():
    """Logical screen size (points) via system_profiler -- no GUI toolkit.

    Avoids tkinter (which aborts with 'Tcl_FindHashEntry on deleted table' when
    mixed with OpenCV's Cocoa window on macOS).
    """
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=6).stdout
        m = re.search(r"UI Looks like:\s*(\d+)\s*x\s*(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"Resolution:\s*(\d+)\s*x\s*(\d+)", out)
        if m:
            w, h = int(m.group(1)), int(m.group(2))
            if w >= 2560:                 # Retina physical -> approximate points
                w, h = w // 2, h // 2
            return w, h
    except Exception:
        logger.warning("Screen-size detection failed; using default", exc_info=True)
    return 1280, 720


def open_camera():
    """Find all cameras, then open the saved one (else the configured one, else any).

    Returns ``(camera, previews)`` with a thumbnail per found camera index.
    """
    t0 = time.time()
    previews = cameras.probe(range(config.CAMERA_MAX_INDEX))
    logger.info("Found cameras %s in %.1f s", sorted(previews), time.time() - t0)
    saved = cameras.load_camera_index(config.SETTINGS_FILE)
    first = cameras.choose_index(saved, config.CAMERA_INDEX, list(previews))
    camera = cameras.Camera()
    if not camera.open_first(cameras.candidate_order(first, config.CAMERA_INDEX)):
        logger.error("Could not open any camera (tried indices 0-%d)",
                     config.CAMERA_MAX_INDEX - 1)
        print("ERROR: Could not open any camera.", file=sys.stderr)
        sys.exit(1)
    return camera, previews


# --------------------------------------------------------------------------- #
# Scaling + layout helpers
# --------------------------------------------------------------------------- #
def _s(px):
    return int(round(px * config.SCALE))


def crop_to_aspect(img, aspect):
    """Centre-crop an image to the given width/height aspect (no distortion)."""
    h, w = img.shape[:2]
    if w / h > aspect:                       # too wide -> crop the sides
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    nh = int(round(w / aspect))              # too tall -> crop top/bottom
    y0 = (h - nh) // 2
    return img[y0:y0 + nh, :]


def row_targets(w, h, n, y_frac=0.46, r=None):
    """n evenly spread circles in a horizontal row: list of (cx, cy, r)."""
    r = _s(config.SELECT_RADIUS if r is None else r)
    y = int(h * y_frac)
    return [(int(w * (i + 1) / (n + 1)), y, r) for i in range(n)]


def quit_target(w, h):
    return (int(w * 0.5), int(h * 0.80), _s(40))


def menu_start_target(w, h):
    return (int(w * 0.5), int(h * 0.38), _s(config.CONFIRM_RADIUS))


def color_swatches(w, h):
    y = int(h * 0.52)
    r = _s(34)
    xs = [int(w * 0.40), int(w * 0.5), int(w * 0.60)]
    return [(xs[i], y, r) for i in range(len(COLORS))]


def settings_button(w, h):
    return (_s(20), h - _s(70), _s(260), _s(48))


def background_button(w, h):
    """Bottom-left button stacked above 'Stiftfarbe'; cycles the background mode."""
    return (_s(20), h - _s(70) - _s(58), _s(260), _s(48))


def handle_background_click(click, mode, w, h):
    """Mouse click on the background button cycles the mode; else returns it unchanged."""
    if _in_rect(click, background_button(w, h)):
        return background.next_mode(mode)
    return mode


def initial_state(identifier):
    """Start with the face scan -- unless face identification is unavailable."""
    return STATE_FACE_SCAN if identifier is not None else STATE_MENU


def player_target(w, h):
    """'Spieler wechseln' circle, right of the Start circle."""
    return (int(w * 0.82), int(h * 0.38), _s(55))


def face_circle(w, h):
    return (w // 2, int(h * 0.48), _s(config.FACE_SCAN_RADIUS))


def guest_target(w, h):
    """'Als Gast spielen' circle on the face-scan screen (bottom left, the camera
    button sits bottom right)."""
    return (int(w * 0.12), int(h * 0.80), _s(45))


def camera_button(w, h):
    """Bottom-right counterpart of the pen-colour button."""
    return (w - _s(20) - _s(200), h - _s(70), _s(200), _s(48))


def show_camera_button(available):
    return len(available) > 1


def reference_still_valid(reference_camera, camera_index):
    """The empty-background reference only fits the camera it was taken with."""
    return reference_camera == camera_index


def handle_camera_click(click, camera_open, available, cap, w, h):
    """Mouse click for the camera selection (menu and face scan). Returns camera_open.

    A click on the button opens the panel; in the open panel a click on a tile
    switches the camera (and remembers it), any click closes the panel.
    """
    if camera_open:
        for index, tile in zip(available, camera_tiles(w, h, len(available))):
            if _in_rect(click, tile) and index != cap.index:
                if cap.switch(index):
                    cameras.save_camera_index(config.SETTINGS_FILE, index)
                break
        return False
    return show_camera_button(available) and _in_rect(click, camera_button(w, h))


def camera_tiles(w, h, n):
    """n preview tiles (x, y, w, h) in a centred row; shrink to fit narrow screens."""
    gap = _s(16)
    tw = min(_s(cameras.PREVIEW_SIZE[0]), (w - gap * (n + 1)) // n)
    th = tw * cameras.PREVIEW_SIZE[1] // cameras.PREVIEW_SIZE[0]
    x0 = (w - (n * tw + (n - 1) * gap)) // 2
    y = int(h * 0.45)
    return [(x0 + i * (tw + gap), y, tw, th) for i in range(n)]


def _hit(point, target):
    px, py = point
    cx, cy, r = target
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def _in_rect(point, rect):
    x, y, bw, bh = rect
    return x <= point[0] <= x + bw and y <= point[1] <= y + bh


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _dim(frame, alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_ring(frame, center, r, progress, color):
    if progress > 0:
        cv2.ellipse(frame, center, (r, r), -90, 0, int(360 * progress), color,
                    max(3, _s(8)), cv2.LINE_AA)


def _draw_circle_option(frame, target, color, label, active, progress):
    cx, cy, r = target
    cv2.circle(frame, (cx, cy), r, color, max(2, _s(3)), cv2.LINE_AA)
    if active:
        _draw_ring(frame, (cx, cy), r, progress, color)
    draw_text(frame, label, (cx, cy + r + _s(26)), scale=0.75, color=color,
              center=True)


def _draw_pen_marker(frame, point):
    if point is not None:
        cv2.circle(frame, point, _s(10), (255, 255, 255), max(1, _s(2)), cv2.LINE_AA)
        cv2.circle(frame, point, _s(3), (255, 255, 255), -1, cv2.LINE_AA)
    return point is not None


def _draw_settings_button(frame, rect, pen_color, is_open):
    x, y, bw, bh = rect
    bg = (90, 90, 90) if is_open else (55, 55, 55)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), bg, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(frame, (x + _s(20), y + bh // 2), _s(11), pen_color.display, -1,
               cv2.LINE_AA)
    draw_text(frame, f"Stiftfarbe: {pen_color.name}",
              (x + _s(40), y + bh // 2 + _s(7)), scale=0.6)


def draw_menu(frame, start_t, quit_t, sbtn, swatches, pen_color, settings_open,
              active, progress, point, player_t=None, player=None):
    _dim(frame)
    h, w = frame.shape[:2]
    draw_text(frame, "FRUIT NINJA CV", (w // 2, int(h * 0.16)), scale=1.4,
              color=(180, 0, 255), thickness=3, center=True)
    _draw_circle_option(frame, start_t, pen_color.display, "Start", active == 0,
                        progress)
    _draw_circle_option(frame, quit_t, (200, 200, 200), "Quit", active == 1,
                        progress)
    if player_t is not None:
        _draw_circle_option(frame, player_t, (200, 200, 200), "Spieler wechseln",
                            active == 2, progress)
        if player is not None:
            px, py, pr = player_t
            draw_text(frame, face_id.display_name(player), (px, py - pr - _s(22)),
                      scale=0.7, color=(0, 215, 255), center=True)
    _draw_settings_button(frame, sbtn, pen_color, settings_open)
    if settings_open:
        _dim(frame, 0.45)
        draw_text(frame, "Stiftfarbe waehlen (Maus-Klick)", (w // 2, int(h * 0.36)),
                  scale=0.9, center=True)
        for i, (cx, cy, r) in enumerate(swatches):
            cv2.circle(frame, (cx, cy), r, COLORS[i].display, -1, cv2.LINE_AA)
            if COLORS[i] is pen_color:
                cv2.circle(frame, (cx, cy), r + _s(6), (255, 255, 255),
                           max(1, _s(2)), cv2.LINE_AA)
            draw_text(frame, COLORS[i].name, (cx, cy + r + _s(28)), scale=0.7,
                      color=COLORS[i].display, center=True)
    if not _draw_pen_marker(frame, point) and not settings_open:
        draw_text(frame, "Stift nicht erkannt", (w // 2, int(h * 0.74)),
                  scale=0.6, color=(80, 80, 255), center=True)


def _draw_camera_button(frame, rect, index, is_open):
    x, y, bw, bh = rect
    bg = (90, 90, 90) if is_open else (55, 55, 55)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), bg, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (200, 200, 200), 1, cv2.LINE_AA)
    draw_text(frame, f"Kamera: {index}", (x + _s(16), y + bh // 2 + _s(7)), scale=0.6)


def _draw_background_button(frame, rect, mode):
    x, y, bw, bh = rect
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (55, 55, 55), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (200, 200, 200), 1, cv2.LINE_AA)
    draw_text(frame, f"Hintergrund: {background.MODE_LABELS.get(mode, mode)}",
              (x + _s(16), y + bh // 2 + _s(7)), scale=0.6)


def draw_camera_panel(frame, tiles, indices, current, previews):
    """Selection panel: one preview tile per camera, the current one framed."""
    _dim(frame, 0.45)
    h, w = frame.shape[:2]
    draw_text(frame, "Kamera waehlen (Maus-Klick)", (w // 2, int(h * 0.36)),
              scale=0.9, center=True)
    for (x, y, tw, th), index in zip(tiles, indices):
        preview = previews.get(index)
        if preview is not None:
            frame[y:y + th, x:x + tw] = cv2.resize(preview, (tw, th))
        else:
            cv2.rectangle(frame, (x, y), (x + tw, y + th), (60, 60, 60), -1)
        color = (255, 255, 255) if index == current else (150, 150, 150)
        cv2.rectangle(frame, (x, y), (x + tw, y + th), color,
                      max(2, _s(4)) if index == current else 1, cv2.LINE_AA)
        draw_text(frame, f"Kamera {index}", (x + tw // 2, y + th + _s(26)), scale=0.65,
                  color=color, center=True)


def draw_difficulty_screen(frame, d_targets, q_target, active, progress, point,
                           color):
    _dim(frame)
    h, w = frame.shape[:2]
    draw_text(frame, "Schwierigkeit waehlen", (w // 2, int(h * 0.20)), scale=1.1,
              center=True)
    draw_text(frame, f"Stift {int(config.DWELL_SECONDS)}s in einen Kreis halten",
              (w // 2, int(h * 0.28)), scale=0.65, center=True)
    for i, t in enumerate(d_targets):
        _draw_circle_option(frame, t, color, LEVELS[i].name, active == i, progress)
    _draw_circle_option(frame, q_target, (200, 200, 200), "Quit",
                        active == len(d_targets), progress)
    _draw_pen_marker(frame, point)


def draw_face_scan(frame, circle, box, inside, progress, greeting, guest_t,
                   guest_active, guest_progress, point):
    """Face-scan screen: circle with progress ring, face box, greeting at the end."""
    h, w = frame.shape[:2]
    cx, cy, r = circle
    _dim(frame, 0.35)
    color = (80, 220, 80) if inside else (230, 230, 230)
    cv2.circle(frame, (cx, cy), r, color, max(2, _s(4)), cv2.LINE_AA)
    _draw_ring(frame, (cx, cy), r + _s(10), progress, (0, 215, 255))
    if box is not None:
        x, y, bw, bh = box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, max(1, _s(2)), cv2.LINE_AA)
    if greeting is not None:
        draw_text(frame, f"Hallo, {face_id.display_name(greeting)}!", (w // 2, int(h * 0.12)),
                  scale=1.4, color=(0, 215, 255), thickness=3, center=True)
        return
    draw_text(frame, "Gesicht in den Kreis halten", (w // 2, int(h * 0.10)), scale=1.1,
              center=True)
    hint = "Stillhalten ..." if inside else "Naeher ran und mittig in den Kreis"
    draw_text(frame, hint, (w // 2, int(h * 0.17)), scale=0.65, center=True)
    _draw_circle_option(frame, guest_t, (200, 200, 200), "Als Gast", guest_active,
                        guest_progress)
    _draw_pen_marker(frame, point)


def draw_name_entry(frame, targets, entry, active, progress, point, color):
    """On-screen keyboard: dwell on a key to type, OK to confirm."""
    _dim(frame, 0.6)
    h, w = frame.shape[:2]
    draw_text(frame, "Neuer Spieler - Name eingeben", (w // 2, int(h * 0.08)), scale=1.0,
              center=True)
    draw_text(frame, f"{entry.text}_", (w // 2, int(h * 0.19)), scale=1.3,
              color=(0, 215, 255), thickness=3, center=True)
    draw_text(frame, f"Stift {config.KEYBOARD_DWELL_SECONDS:g}s auf eine Taste halten  -  "
              f"leer lassen = Spieler-Nummer", (w // 2, int(h * 0.27)), scale=0.55, center=True)
    for i, (cx, cy, r) in enumerate(targets):
        key = keyboard.KEYS[i]
        on = i == active
        cv2.circle(frame, (cx, cy), r, color if on else (200, 200, 200),
                   max(1, _s(2)), cv2.LINE_AA)
        if on:
            _draw_ring(frame, (cx, cy), r, progress, color)
        label = face_id.display_name(key) if len(key) == 1 else key
        draw_text(frame, label, (cx, cy), scale=0.6 if len(key) == 1 else 0.45,
                  center=True)
    _draw_pen_marker(frame, point)


def draw_mask_inset(frame, mask):
    iw, ih = _s(192), _s(108)
    small = cv2.cvtColor(cv2.resize(mask, (iw, ih)), cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    x0, y0 = w - iw - _s(10), _s(10)
    frame[y0:y0 + ih, x0:x0 + iw] = small
    cv2.rectangle(frame, (x0, y0), (x0 + iw, y0 + ih), (255, 255, 255), 1)
    draw_text(frame, "Maske", (x0 + _s(4), y0 + ih - _s(8)), scale=0.5)


GAME_OVER_LABELS = ["Neustart", "Startmenue", "Quit"]


def draw_game_over(frame, game, scores, rank, targets, active, progress, point,
                   color, player):
    _dim(frame, 0.55)
    h, w = frame.shape[:2]
    draw_text(frame, "GAME OVER", (w // 2, int(h * 0.16)), scale=1.6,
              color=(80, 80, 255), thickness=3, center=True)
    draw_text(frame, f"{face_id.display_name(player)}:  {game.score}",
              (w // 2, int(h * 0.26)), scale=1.0, center=True)
    if rank is not None:
        draw_text(frame, f"NEUER HIGHSCORE  (#{rank})!", (w // 2, int(h * 0.33)),
                  scale=0.75, color=(0, 215, 255), center=True)
    draw_text(frame, f"Leaderboard - {game.difficulty.name}",
              (w // 2, int(h * 0.42)), scale=0.7, center=True)
    for i in range(3):
        if i < len(scores):
            line = f"{i + 1}.  {face_id.display_name(scores[i]['name'])}  {scores[i]['score']}"
        else:
            line = f"{i + 1}.  -"
        draw_text(frame, line, (w // 2, int(h * 0.47) + _s(26) * i),
                  scale=0.65, center=True)
    for i, t in enumerate(targets):
        _draw_circle_option(frame, t, color, GAME_OVER_LABELS[i], active == i,
                            progress)
    _draw_pen_marker(frame, point)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    log_path = setup_logging()
    logger.info("Starting Fruit Ninja CV (logfile: %s)", log_path)

    sw, sh = detect_screen_size()
    # Render to a fixed 16:10 aspect (laptop fullscreen area), height from screen.
    config.HEIGHT = sh
    config.WIDTH = int(round(sh * config.ASPECT))
    config.SCALE = sh / config.REF_HEIGHT
    logger.info("Screen %dx%d -> canvas %dx%d, scale %.2f", sw, sh,
                config.WIDTH, config.HEIGHT, config.SCALE)

    cap = None
    try:
        cap, previews = open_camera()
        sounds = SoundPlayer()
        identifier = face_id.create_identifier()
        _run_game_loop(cap, sounds, previews, identifier)
    except SystemExit:
        raise
    except Exception:
        logger.exception("Unhandled error -- see the logfile for the traceback")
        raise
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Shut down cleanly")


def _run_game_loop(cap, sounds, previews=None, identifier=None):
    tracker = MarkerTracker()
    blade = Blade()
    dwell = DwellTracker(config.DWELL_SECONDS)
    game = None
    difficulty = None
    pen_color = COLORS[0]            # Pink by default -- only this colour is tracked
    tracker.set_color(pen_color)
    scores, rank = [], None
    player = face_id.GUEST
    faces_dir = os.path.join(face_id.BASE_DIR, config.FACES_DIR)
    scanner = face_id.FaceScanner()
    greet_until = None             # set once the player is decided (greeting shown)
    new_crops = []                 # face crops of a new player until the name is typed
    entry = keyboard.NameEntry()
    key_dwell = DwellTracker(config.KEYBOARD_DWELL_SECONDS)
    typed_key = None               # key from the physical keyboard (name entry)
    show_mask = False
    settings_open = False
    camera_open = False
    previews = dict(previews or {})  # thumbnails of the cameras found at startup
    frame_failures = 0
    click = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click[0] = (x, y)

    state = initial_state(identifier)
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    if config.FULLSCREEN:
        cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        try:
            cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO,
                                  cv2.WINDOW_FREERATIO)
        except Exception:
            logger.warning("WND_PROP_ASPECT_RATIO not supported", exc_info=True)
    cv2.setMouseCallback(config.WINDOW_NAME, on_mouse)

    # Space background: load the backdrop. The reference is NOT captured
    # automatically -- the player stands in frame to use the menu, so an auto
    # capture would bake their silhouette into the background. They press 'b'
    # (then step out during the countdown) to grab a clean empty background.
    space_img = background.load_space((config.WIDTH, config.HEIGHT))
    suit_smoother = suit.RectSmoother()
    reference = None
    reference_camera = None          # camera the reference was taken with
    bg_capture_at = None
    bg_mode = cameras.load_background_mode(config.SETTINGS_FILE) or config.BG_MODE
    if bg_mode not in background.MODES:
        bg_mode = config.BG_MODE
    logger.info("Background mode: %s", bg_mode)
    # YOLO body segmenter for the "yolo" mode (downloaded on first use); None if
    # the model is unavailable, then the mode falls back like "off".
    segmenter = yolo_seg.create_segmenter() if config.YOLO_ENABLED else None

    while True:
        ok, frame = cap.read()
        if not ok:
            frame_failures += 1
            if frame_failures >= config.MAX_FRAME_FAILURES:
                logger.error("Lost camera after %d consecutive failed reads -- aborting",
                             frame_failures)
                print("ERROR: Lost camera feed.", file=sys.stderr)
                break
            cv2.waitKey(10)
            continue
        frame_failures = 0

        frame = crop_to_aspect(frame, config.WIDTH / config.HEIGHT)
        frame = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
        if config.MIRROR:
            frame = cv2.flip(frame, 1)
        clean = frame.copy()

        # --- space background: capture the reference, then cut the player out ---
        if bg_capture_at is not None and time.monotonic() >= bg_capture_at:
            reference = clean.copy()
            reference_camera = cap.index
            bg_capture_at = None
            logger.info("Captured background reference for space compositing")
        if reference is not None and not reference_still_valid(reference_camera, cap.index):
            reference = None
            logger.info("Camera changed -- space background off until 'b' is pressed again")
        if config.BG_ENABLED and space_img is not None and bg_mode != background.MODE_OFF:
            mask = None
            if bg_mode == background.MODE_GREENSCREEN:
                mask = background.greenscreen_mask(clean, scale=config.GREEN_SCALE)
            elif bg_mode == background.MODE_YOLO and segmenter is not None:
                mask = segmenter.segment(clean)      # person mask, no green screen needed
            elif bg_mode == background.MODE_NORMAL and reference is not None:
                mask = background.foreground_mask(clean, reference)
            if mask is not None:
                frame = background.composite_space(frame, mask, space_img)
                if config.SUIT_ENABLED:
                    # place the helmet at the face (if one is seen) -- detection only, ~3 ms
                    face_box = (identifier.largest_face_box(clean, config.SUIT_FACE_DETECT_WIDTH)
                                if identifier is not None else None)
                    # the whole body becomes the suit; only the pen stays visible
                    frame = suit.dress(frame, mask, face_box, tracker.last_mask,
                                       time.monotonic(), suit_smoother)

        quit_requested = False

        if state == STATE_MENU:
            start_t = menu_start_target(config.WIDTH, config.HEIGHT)
            quit_t = quit_target(config.WIDTH, config.HEIGHT)
            sbtn = settings_button(config.WIDTH, config.HEIGHT)
            swatches = color_swatches(config.WIDTH, config.HEIGHT)

            cbtn = camera_button(config.WIDTH, config.HEIGHT)
            previews[cap.index] = cv2.resize(clean, cameras.PREVIEW_SIZE)
            available = sorted(previews)
            cam_tiles = camera_tiles(config.WIDTH, config.HEIGHT, len(available))

            if click[0] is not None:
                was_open = camera_open
                camera_open = handle_camera_click(click[0], camera_open, available, cap,
                                                  config.WIDTH, config.HEIGHT)
                if was_open or camera_open:  # click belonged to the camera selection
                    settings_open = False
                    dwell.reset()
                    click[0] = None

            if click[0] is not None:
                new_mode = handle_background_click(click[0], bg_mode,
                                                   config.WIDTH, config.HEIGHT)
                if new_mode != bg_mode:          # click cycled the background mode
                    bg_mode = new_mode
                    cameras.save_background_mode(config.SETTINGS_FILE, bg_mode)
                    logger.info("Background mode set to %s (mouse)", bg_mode)
                    settings_open = False
                    dwell.reset()
                    click[0] = None

            if click[0] is not None:
                if _in_rect(click[0], sbtn):
                    settings_open = not settings_open
                    dwell.reset()
                elif settings_open:
                    picked = False
                    for i, sw_t in enumerate(swatches):
                        if _hit(click[0], sw_t):
                            pen_color = COLORS[i]
                            tracker.set_color(pen_color)
                            logger.info("Pen colour set to %s (mouse)", pen_color.name)
                            picked = True
                            break
                    settings_open = False   # any click in the panel closes it
                click[0] = None

            point = tracker.track(clean)
            player_t = player_target(config.WIDTH, config.HEIGHT) if identifier else None
            targets = [start_t, quit_t] + ([player_t] if player_t else [])
            if settings_open or camera_open:
                active, progress, confirmed = None, 0.0, None
            else:
                active, progress, confirmed = dwell.update(point, targets)
            draw_menu(frame, start_t, quit_t, sbtn, swatches, pen_color,
                      settings_open, active, progress, point, player_t,
                      player if identifier else None)
            if not settings_open and not camera_open:
                _draw_background_button(frame, background_button(config.WIDTH, config.HEIGHT),
                                        bg_mode)
            if show_camera_button(available):
                _draw_camera_button(frame, cbtn, cap.index, camera_open)
            if camera_open:
                draw_camera_panel(frame, cam_tiles, available, cap.index, previews)
            if confirmed == 0:              # Start
                logger.info("Start (colour %s, player %s)", pen_color.name, player)
                dwell.reset()
                state = STATE_DIFFICULTY
            elif confirmed == 1:            # Quit
                quit_requested = True
            elif confirmed == 2:            # Spieler wechseln
                logger.info("Switch player")
                dwell.reset()
                scanner.reset()
                greet_until = None
                state = STATE_FACE_SCAN

        elif state == STATE_FACE_SCAN:
            circle = face_circle(config.WIDTH, config.HEIGHT)
            guest_t = guest_target(config.WIDTH, config.HEIGHT)
            # camera selection here too: with the wrong camera nobody could be scanned
            previews[cap.index] = cv2.resize(clean, cameras.PREVIEW_SIZE)
            available = sorted(previews)
            if click[0] is not None:
                camera_open = handle_camera_click(click[0], camera_open, available, cap,
                                                  config.WIDTH, config.HEIGHT)
                click[0] = None
                scanner.reset()          # restart the scan with the (new) camera
            name, score, box, crop = identifier.scan(clean)
            inside = face_id.face_in_circle(box, circle) and not camera_open
            point = tracker.track(clean)
            g_active, g_progress, g_confirmed = dwell.update(point, [guest_t])
            if camera_open:
                g_confirmed = None
            if greet_until is None:
                result = scanner.update(inside, name, crop)
                if g_confirmed is not None:
                    result = ("known", face_id.GUEST)
                if result is not None and result[0] == "known":
                    player = result[1]
                    greet_until = time.time() + config.FACE_GREETING_SECONDS
                    logger.info("Player recognised: %s", player)
                elif result is not None:
                    new_crops = result[1]
                    entry = keyboard.NameEntry()
                    key_dwell.reset()
                    typed_key = None
                    logger.info("New face recorded (%d crops) -- asking for the name",
                                len(new_crops))
                    state = STATE_NAME_ENTRY
            if state == STATE_FACE_SCAN:
                draw_face_scan(frame, circle, box, inside, scanner.progress,
                               player if greet_until is not None else None, guest_t,
                               g_active == 0, g_progress, point)
                if show_camera_button(available):
                    _draw_camera_button(frame, camera_button(config.WIDTH, config.HEIGHT),
                                        cap.index, camera_open)
                if camera_open:
                    draw_camera_panel(frame, camera_tiles(config.WIDTH, config.HEIGHT,
                                                          len(available)),
                                      available, cap.index, previews)
                if greet_until is not None and time.time() >= greet_until:
                    dwell.reset()
                    state = STATE_MENU

        elif state == STATE_NAME_ENTRY:
            point = tracker.track(clean)
            k_targets = keyboard.layout(config.WIDTH, config.HEIGHT)
            active, progress, confirmed = key_dwell.update(point, k_targets)
            pressed = typed_key
            typed_key = None
            if confirmed is not None:
                pressed = keyboard.KEYS[confirmed]
                key_dwell.reset()        # hold again to repeat the same key
            if pressed is not None and entry.press(pressed):
                player = entry.result() or face_id.default_player_name(identifier.persons)
                identifier.enroll(faces_dir, player, new_crops)
                new_crops = []
                scanner.reset()
                greet_until = time.time() + config.FACE_GREETING_SECONDS
                state = STATE_FACE_SCAN  # shows the greeting, then the menu
            else:
                draw_name_entry(frame, k_targets, entry, active, progress, point,
                                pen_color.display)

        elif state == STATE_DIFFICULTY:
            point = tracker.track(clean)
            d_targets = row_targets(config.WIDTH, config.HEIGHT, len(LEVELS))
            q_target = quit_target(config.WIDTH, config.HEIGHT)
            active, progress, confirmed = dwell.update(point, d_targets + [q_target])
            draw_difficulty_screen(frame, d_targets, q_target, active, progress,
                                   point, pen_color.display)
            if confirmed is not None:
                if confirmed == len(d_targets):
                    quit_requested = True
                else:
                    difficulty = LEVELS[confirmed]
                    game = Game(config.WIDTH, config.HEIGHT, difficulty, sounds)
                    sounds.play("combo")
                    logger.info("Difficulty selected: %s", difficulty.name)
                    blade.reset()
                    dwell.reset()
                    state = STATE_PLAY

        elif state == STATE_PLAY:
            point = tracker.track(clean)
            blade.add(point)
            game.update(blade)
            game.draw(frame)
            blade.draw(frame)
            game.draw_hud(frame)
            game.draw_combo(frame)
            if game.game_over:
                scores, rank = highscore.add(config.HIGHSCORE_FILE,
                                             game.difficulty.name, game.score, player)
                sounds.play("over")
                logger.info("Game over -- %s, %s scored %d (rank: %s)",
                            game.difficulty.name, player, game.score, rank)
                dwell.reset()
                state = STATE_GAME_OVER

        elif state == STATE_GAME_OVER:
            point = tracker.track(clean)
            targets = row_targets(config.WIDTH, config.HEIGHT, 3, y_frac=0.80, r=50)
            active, progress, confirmed = dwell.update(point, targets)
            game.draw(frame)
            game.draw_hud(frame)
            draw_game_over(frame, game, scores, rank, targets, active, progress,
                           point, pen_color.display, player)
            if confirmed == 0:              # Neustart
                logger.info("Restart requested (%s)", game.difficulty.name)
                game.reset()
                blade.reset()
                rank = None
                dwell.reset()
                state = STATE_PLAY
            elif confirmed == 1:            # Startmenue
                logger.info("Back to start menu")
                rank = None
                dwell.reset()
                state = STATE_MENU
            elif confirmed == 2:            # Quit
                quit_requested = True

        if bg_capture_at is not None:
            secs = max(0, int(bg_capture_at - time.monotonic()) + 1)
            draw_text(frame, f"Aus dem Bild gehen: {secs}",
                      (config.WIDTH // 2, int(config.HEIGHT * 0.12)),
                      scale=1.0, center=True)
        elif (config.BG_ENABLED and space_img is not None and reference is None
              and bg_mode == background.MODE_NORMAL          # greenscreen needs no reference
              and state != STATE_NAME_ENTRY):        # there 'b' types a letter
            draw_text(frame, "Taste b: Weltall-Hintergrund",
                      (config.WIDTH // 2, int(config.HEIGHT * 0.96)),
                      scale=0.5, center=True)
        elif bg_mode == background.MODE_YOLO and segmenter is None:
            draw_text(frame, "YOLO-Modell nicht verfuegbar", (config.WIDTH // 2,
                      int(config.HEIGHT * 0.96)), scale=0.5, color=(80, 80, 255),
                      center=True)

        if show_mask and tracker.last_mask is not None:
            cv2.imshow("mask (debug)",
                       cv2.cvtColor(tracker.last_mask, cv2.COLOR_GRAY2BGR))

        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if state == STATE_NAME_ENTRY and keyboard.from_keycode(key) is not None:
            typed_key = keyboard.from_keycode(key)   # letters type here instead of q/d
        elif quit_requested or key == ord("q"):
            logger.info("Quit requested")
            break
        elif key == ord("d"):
            show_mask = not show_mask
            logger.info("Debug mask %s", "on" if show_mask else "off")
            if not show_mask:
                cv2.destroyWindow("mask (debug)")
        elif key == ord("b") and space_img is not None:
            bg_capture_at = time.monotonic() + config.BG_COUNTDOWN_SECONDS
            logger.info("Background (re)capture armed -- %.1fs countdown",
                        config.BG_COUNTDOWN_SECONDS)


if __name__ == "__main__":
    main()
