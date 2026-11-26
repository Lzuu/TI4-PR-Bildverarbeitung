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
import numpy as np

import config
import background
import suit
import lightsaber
import yolo_seg
from contextlib import nullcontext
from inference_worker import LatestWorker
from async_perception import AsyncSegmenter, AsyncPoseInfer
import pose_tracking
from suit_rig import SuitRenderer
from marker_tracker import MarkerTracker
from blade import Blade
from challenge import ChallengeGame
from game import Game
from motion import StrokeTracker
from utils import draw_text
from logging_config import setup_logging
from difficulty import LEVELS
from colors import COLORS, DEFAULT as DEFAULT_PEN, GRUEN
from dwell import DwellTracker
from sound import SoundPlayer
import cameras
import face_id
import highscore
import perf
import keyboard

logger = logging.getLogger(__name__)

STATE_FACE_SCAN = "face_scan"
STATE_NAME_ENTRY = "name_entry"
STATE_MENU = "menu"
STATE_DIFFICULTY = "difficulty"
STATE_PLAY = "play"
STATE_GAME_OVER = "game_over"


DEFAULT_SCREEN = (1280, 720)


def _windows_user32():
    import ctypes
    return ctypes.windll.user32


def detect_screen_size(platform=None):
    """Logical screen size of the primary display -- no GUI toolkit.

    Windows: ``GetSystemMetrics`` (logical pixels, i.e. after display scaling,
    the same units the OpenCV window uses; #41, CR-08). macOS: system_profiler
    (tkinter aborts with 'Tcl_FindHashEntry on deleted table' when mixed with
    OpenCV's Cocoa window). Anything else or any failure: ``DEFAULT_SCREEN``.
    """
    platform = platform or sys.platform
    try:
        if platform == "win32":
            user32 = _windows_user32()
            w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return w, h
        elif platform == "darwin":
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
        return DEFAULT_SCREEN
    logger.warning("Screen size unknown on %s; using default %dx%d", platform, *DEFAULT_SCREEN)
    return DEFAULT_SCREEN


def canvas_size(sw, sh, platform=None):
    """Render size ``(width, height)`` for a screen of ``sw`` x ``sh``.

    macOS: full screen height at the 16:10 usable fullscreen area (``ASPECT``).
    Elsewhere the canvas takes the screen's own aspect, so the fullscreen
    stretch does not distort (#41, CR-08), and its height is capped at
    ``MAX_CANVAS_HEIGHT`` because the frame time grows with the pixel count.
    """
    platform = platform or sys.platform
    if platform == "darwin":
        return int(round(sh * config.ASPECT)), sh
    h = min(sh, config.MAX_CANVAS_HEIGHT)
    return int(round(h * sw / sh)), h


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


def pen_allowed(pen, bg_mode):
    """A green pen cannot be told apart from a green screen (#41, CR-04)."""
    return not (bg_mode == background.MODE_GREENSCREEN and pen is GRUEN)


def pen_for_mode(pen, bg_mode):
    """The pen colour to use in ``bg_mode``: ``pen`` if allowed, else the default."""
    return pen if pen_allowed(pen, bg_mode) else DEFAULT_PEN


PEN_HINT = "Gruener Stift geht nicht mit Greenscreen"
NOT_SAVED_HINT = "Gesicht konnte nicht gespeichert werden"


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


MODE_CLASSIC = "classic"
MODE_CHALLENGE = "challenge"
MENU_START, MENU_QUIT, MENU_CHALLENGE, MENU_PLAYER = 0, 1, 2, 3    # order of the dwell targets


def challenge_target(w, h):
    """'Challenge' circle, left of the Start circle (the player circle is on the right)."""
    return (int(w * 0.18), int(h * 0.38), _s(80))


def menu_targets(w, h, with_player):
    """The dwell targets of the start menu, in the order of the MENU_* indices."""
    targets = [menu_start_target(w, h), quit_target(w, h), challenge_target(w, h)]
    if with_player:
        targets.append(player_target(w, h))
    return targets


def new_game(mode, difficulty, sounds):
    """The game for the chosen mode (the start menu chooses it, #33)."""
    if mode == MODE_CHALLENGE:
        return ChallengeGame(config.WIDTH, config.HEIGHT, difficulty, sounds)
    return Game(config.WIDTH, config.HEIGHT, difficulty, sounds)


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
              active, progress, point, player_t=None, player=None, blocked=(),
              challenge_t=None):
    _dim(frame)
    h, w = frame.shape[:2]
    draw_text(frame, "FRUIT NINJA CV", (w // 2, int(h * 0.16)), scale=1.4,
              color=(180, 0, 255), thickness=3, center=True)
    _draw_circle_option(frame, start_t, pen_color.display, "Start", active == MENU_START,
                        progress)
    _draw_circle_option(frame, quit_t, (200, 200, 200), "Quit", active == MENU_QUIT,
                        progress)
    if challenge_t is not None:
        _draw_circle_option(frame, challenge_t, (0, 215, 255), "Challenge",
                            active == MENU_CHALLENGE, progress)
    if player_t is not None:
        _draw_circle_option(frame, player_t, (200, 200, 200), "Spieler wechseln",
                            active == MENU_PLAYER, progress)
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
            if COLORS[i] in blocked:             # e.g. green while greenscreen is on
                d = int(r * 0.8)
                cv2.line(frame, (cx - d, cy - d), (cx + d, cy + d), (60, 60, 60), max(2, _s(5)))
                cv2.line(frame, (cx - d, cy + d), (cx + d, cy - d), (60, 60, 60), max(2, _s(5)))
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


def draw_name_entry(frame, targets, entry, active, progress, point, color, message=None):
    """On-screen keyboard: dwell on a key to type, OK to confirm. ``message``: why the
    typed name was refused (shown in red)."""
    _dim(frame, 0.6)
    h, w = frame.shape[:2]
    draw_text(frame, "Neuer Spieler - Name eingeben", (w // 2, int(h * 0.08)), scale=1.0,
              center=True)
    draw_text(frame, f"{entry.text}_", (w // 2, int(h * 0.19)), scale=1.3,
              color=(0, 215, 255), thickness=3, center=True)
    draw_text(frame, f"Stift {config.KEYBOARD_DWELL_SECONDS:g}s auf eine Taste halten  -  "
              f"leer lassen = Spieler-Nummer", (w // 2, int(h * 0.27)), scale=0.55, center=True)
    if message:
        draw_text(frame, message, (w // 2, int(h * 0.32)), scale=0.7, color=(80, 80, 255),
                  center=True)
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


def draw_no_camera():
    """Black screen with a notice while no camera is left (#69)."""
    canvas = np.zeros((config.HEIGHT, config.WIDTH, 3), np.uint8)
    draw_text(canvas, "Kamera nicht verfuegbar - neuer Versuch ...",
              (config.WIDTH // 2, config.HEIGHT // 2), scale=0.9, center=True)
    draw_text(canvas, "Taste q: Beenden", (config.WIDTH // 2, int(config.HEIGHT * 0.58)),
              scale=0.6, color=(200, 200, 200), center=True)
    return canvas


GAME_OVER_LABELS = ["Neustart", "Startmenue", "Quit"]


def draw_game_over(frame, game, scores, rank, targets, active, progress, point,
                   color, player, saved=True):
    _dim(frame, 0.55)
    h, w = frame.shape[:2]
    draw_text(frame, game.title, (w // 2, int(h * 0.16)), scale=1.6,
              color=(80, 80, 255), thickness=3, center=True)
    draw_text(frame, f"{face_id.display_name(player)}:  {game.result_text()}",
              (w // 2, int(h * 0.26)), scale=1.0, center=True)
    if rank is not None:
        draw_text(frame, f"NEUER HIGHSCORE  (#{rank})!", (w // 2, int(h * 0.33)),
                  scale=0.75, color=(0, 215, 255), center=True)
    draw_text(frame, f"Leaderboard - {game.board_name}",
              (w // 2, int(h * 0.42)), scale=0.7, center=True)
    if not saved:                    # honest storage status (#41, CR-07)
        draw_text(frame, "Highscore konnte nicht gespeichert werden", (w // 2, int(h * 0.37)),
                  scale=0.55, color=(80, 80, 255), center=True)
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
    config.WIDTH, config.HEIGHT = canvas_size(sw, sh)
    config.SCALE = config.HEIGHT / config.REF_HEIGHT
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


def draw_perf_overlay(frame, lines):
    """Frame-time statistics (key 'p') in the top-right corner (#42, PERF-06)."""
    if not lines:
        lines = ["Messung laeuft ..."]
    x = config.WIDTH - _s(330)
    for i, line in enumerate(lines[:12]):
        draw_text(frame, line, (x, _s(28) + i * _s(22)), scale=0.45)


class Frame:
    """What one pass of the loop knows about the current camera picture (#73, CR-23).

    ``frame`` is what is drawn on and finally shown; ``clean`` is the untouched camera
    picture -- detection always runs on ``clean``, never on ``frame``.
    """

    def __init__(self, dt, now):
        self.dt = dt                  # real seconds since the previous frame
        self.now = now                # time.monotonic() right after the frame was read
        self.frame = None
        self.clean = None
        self.point = None             # pen tip, or None when it is not seen
        self.pen_mask = None
        self.face_box = None
        self.face_checked = False     # a face detection already ran this frame (even if empty)
        self.scan_result = None       # (name, score, box, crop) on the face-scan screen
        self.saber_hands = {}         # gloves the suit draws in THIS frame, never older (#46)
        self.quit_requested = False


_RETRY, _STOP = object(), object()    # outcomes of reading a camera frame that failed


class GameLoop:
    """The game loop and everything it remembers between frames (#73, CR-23).

    ``run()`` does one pass per camera frame: read the picture, find the pen, cut the player
    out and dress them, let the current screen update and draw, then the lightsaber, the
    display and the keys. Each screen (face scan, name entry, menu, difficulty, play, game
    over) is one ``_<name>_screen`` method; ``self.state`` says which one is current.
    """

    SCREENS = {                          # state -> the method that updates and draws it
        STATE_FACE_SCAN: "_face_scan_screen",
        STATE_NAME_ENTRY: "_name_entry_screen",
        STATE_MENU: "_menu_screen",
        STATE_DIFFICULTY: "_difficulty_screen",
        STATE_PLAY: "_play_screen",
        STATE_GAME_OVER: "_game_over_screen",
    }

    def __init__(self, cap, sounds, previews=None, identifier=None, worker=None):
        self.cap = cap
        self.worker = worker                 # latest-only inference worker, or None
        self.sounds = sounds
        self.identifier = identifier
        self.tracker = MarkerTracker(point_mode=config.PEN_POINT)   # tip = lightsaber tip, cuts
        self.blade = Blade()
        self.motion = StrokeTracker(config.WIDTH)    # timestamped pen track for direction cuts
        self.play_mode = MODE_CLASSIC                # chosen in the start menu
        self.dwell = DwellTracker(config.DWELL_SECONDS)
        self.game = None
        self.difficulty = None
        self.pen_color = COLORS[0]       # Pink by default -- only this colour is tracked
        self.tracker.set_color(self.pen_color)
        self.hint_text, self.hint_until = None, 0.0   # short on-screen notice (e.g. pen/mode)
        self.scores, self.rank = [], None
        self.score_saved = True
        self.player = face_id.GUEST
        self.faces_dir = os.path.join(face_id.BASE_DIR, config.FACES_DIR)
        self.scanner = face_id.FaceScanner()
        self.greet_until = None          # set once the player is decided (greeting shown)
        self.new_crops = []              # face crops of a new player until the name is typed
        self.entry = keyboard.NameEntry()
        self.key_dwell = DwellTracker(config.KEYBOARD_DWELL_SECONDS)
        self.typed_key = None            # key from the physical keyboard (name entry)
        self.show_mask = False
        self.settings_open = False
        self.camera_open = False
        self.previews = dict(previews or {})   # thumbnails of the cameras found at startup
        self.frame_failures = 0
        self.next_reopen, self.no_camera_since = 0.0, None   # while no camera is left (#69)
        self.name_message, self.name_message_until = None, 0.0   # why a name was refused (#70)
        self.click = None                # last mouse click (x, y) not handled yet
        self.stats = perf.FrameStats(interval=config.PERF_LOG_INTERVAL, csv_path=config.PERF_CSV)
        self.show_perf = False

        self.state = initial_state(identifier)
        self._open_window()

        # Space background: load the backdrop. The reference is NOT captured
        # automatically -- the player stands in frame to use the menu, so an auto
        # capture would bake their silhouette into the background. They press 'b'
        # (then step out during the countdown) to grab a clean empty background.
        self.space_img = background.load_space((config.WIDTH, config.HEIGHT))
        self.suit_renderer = SuitRenderer() if config.SUIT_ENABLED else None
        self.pose_tracker = ((pose_tracking.PoseTracker(AsyncPoseInfer(worker)) if worker else
                              pose_tracking.create_tracker()) if config.SUIT_ENABLED else None)
        self.saber = lightsaber.SaberTracker()
        self.reference = None
        self.reference_camera = None     # camera the reference was taken with
        self.bg_capture_at = None
        self.bg_mode = cameras.load_background_mode(config.SETTINGS_FILE) or config.BG_MODE
        if self.bg_mode not in background.MODES:
            self.bg_mode = config.BG_MODE
        logger.info("Background mode: %s", self.bg_mode)
        # YOLO loads on first use. In async mode the worker owns both loading and
        # inference; until a fresh result is ready the camera view stays usable.
        self.yolo = ((AsyncSegmenter(worker) if worker else yolo_seg.LazySegmenter())
                     if config.YOLO_ENABLED else None)
        self.mode_session = 0            # +1 on every background mode change (YOLO cache source)

        self.last_frame_t = time.monotonic()

    def _open_window(self):
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        if config.FULLSCREEN:
            cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
            try:
                cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO,
                                      cv2.WINDOW_FREERATIO)
            except Exception:
                logger.warning("WND_PROP_ASPECT_RATIO not supported", exc_info=True)
        cv2.setMouseCallback(config.WINDOW_NAME, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)

    # -- the loop ---------------------------------------------------------------------
    def run(self):
        while True:
            self.stats.start_frame()
            f = self._next_frame()
            if f is _STOP:
                break
            if f is _RETRY:
                continue
            self._find_pen_and_face(f)
            self._cut_out_player(f)
            self._run_screen(f)
            self._draw_status_text(f)
            self._draw_debug_mask()
            self._draw_saber(f)
            key = self._show(f)
            if self._handle_key(f, key):
                break

    def _next_frame(self):
        """The next camera picture as a ``Frame`` (cropped, scaled, mirrored), or
        ``_RETRY`` / ``_STOP`` when there is none."""
        with self.stats.stage("read"):
            ok, image = self.cap.read()
        if not ok:
            return self._read_failed()
        self.frame_failures = 0
        now = time.monotonic()
        dt, self.last_frame_t = now - self.last_frame_t, now   # real time since the last frame
        f = Frame(dt, now)
        with self.stats.stage("prep"):
            image = crop_to_aspect(image, config.WIDTH / config.HEIGHT)
            image = cv2.resize(image, (config.WIDTH, config.HEIGHT))
            if config.MIRROR:
                image = cv2.flip(image, 1)
            f.frame = image
            f.clean = image.copy()
        return f

    def _read_failed(self):
        self.stats.count("read_fail")
        self.frame_failures += 1
        if not self.cap.is_open:
            # No camera at all (a switch and its rollback failed): try to open one again
            # at intervals instead of counting failed reads up to the abort (#69).
            now = time.monotonic()
            if self.no_camera_since is None:
                self.no_camera_since = now
                logger.warning("No camera -- trying to open one every %.1f s",
                               config.CAMERA_REOPEN_INTERVAL)
            if now >= self.next_reopen:
                self.next_reopen = now + config.CAMERA_REOPEN_INTERVAL
                if self.cap.reopen():
                    logger.info("Camera is back (index %s)", self.cap.index)
                    self.no_camera_since, self.frame_failures = None, 0
                    return _RETRY
            if now - self.no_camera_since >= config.CAMERA_LOST_TIMEOUT:
                logger.error("No camera for %.0f s -- aborting", now - self.no_camera_since)
                print("ERROR: Lost camera feed.", file=sys.stderr)
                return _STOP
            cv2.imshow(config.WINDOW_NAME, draw_no_camera())
            if cv2.waitKey(100) & 0xFF == ord("q"):
                return _STOP
            return _RETRY
        if self.frame_failures >= config.MAX_FRAME_FAILURES:
            logger.error("Lost camera after %d consecutive failed reads -- aborting",
                         self.frame_failures)
            print("ERROR: Lost camera feed.", file=sys.stderr)
            return _STOP
        cv2.waitKey(10)
        return _RETRY

    # -- perception ------------------------------------------------------------------------
    def _find_pen_and_face(self, f):
        # perception first: one pen measurement per frame, used by suit, game and
        # drawing alike -- no stale mask from the previous frame (#41, CR-06)
        with self.stats.stage("track"):
            f.point = self.tracker.track(f.clean)
        f.pen_mask = self.tracker.pen_mask   # only the pen, not other objects of its colour
        # One face detection per frame, shared by suit, saber and the player scan
        # (#42, PERF-05). In FACE_SCAN the scan's own detection (with recognition)
        # is the one; otherwise a cheap detection runs at most once, on demand.
        if self.state == STATE_FACE_SCAN and self.identifier is not None:
            with self.stats.stage("face"):
                f.scan_result = self.identifier.scan(f.clean)
            f.face_box, f.face_checked = f.scan_result[2], True

    def _cut_out_player(self, f):
        """Space background: capture the reference, then cut the player out and dress them."""
        if self.bg_capture_at is not None and time.monotonic() >= self.bg_capture_at:
            self.reference = f.clean.copy()
            self.reference_camera = self.cap.index
            self.bg_capture_at = None
            logger.info("Captured background reference for space compositing")
        if self.reference is not None and not reference_still_valid(self.reference_camera,
                                                                    self.cap.index):
            self.reference = None
            logger.info("Camera changed -- space background off until 'b' is pressed again")
        if (config.BG_ENABLED and self.space_img is not None
                and self.bg_mode != background.MODE_OFF):
            mask = self._player_mask(f)
            if mask is not None:
                with self.stats.stage("composite"):
                    f.frame = background.composite_space(
                        f.frame, mask, self.space_img,
                        feather=1.5 if config.SUIT_ENABLED else 0.)
                if config.SUIT_ENABLED:
                    self._dress_player(f, mask)

    def _player_mask(self, f):
        t_stage = time.perf_counter()
        mask = None
        if self.bg_mode == background.MODE_GREENSCREEN:
            mask = background.greenscreen_mask(f.clean, scale=config.GREEN_SCALE)
        elif self.bg_mode == background.MODE_YOLO and self.yolo is not None:
            self.yolo.request()                       # first use: start loading
            segmenter = self.yolo if self.worker else self.yolo.get()
            if segmenter is not None:
                # person mask, no green screen needed; the cache is bound to the
                # current camera and mode session (#41, CR-10)
                if self.worker:
                    mask = segmenter.segment(f.clean, source=(self.cap.index, self.mode_session),
                                             captured_at=f.now)
                else:
                    mask = segmenter.segment(f.clean, source=(self.cap.index, self.mode_session))
                if segmenter.last_inference_seconds is not None:
                    self.stats.add("yolo_infer", segmenter.last_inference_seconds)
                self.stats.maximum("mask_age_ms", segmenter.mask_age * 1000.0)
                if segmenter.failed:
                    self.yolo.mark_failed()           # off until chosen again (CR-09)
        elif self.bg_mode == background.MODE_NORMAL and self.reference is not None:
            mask = background.foreground_mask(f.clean, self.reference)
        self.stats.add("segment", time.perf_counter() - t_stage)
        return mask

    def _dress_player(self, f, mask):
        # place the helmet at the face (if one is seen) -- detection only, ~3 ms
        if not f.face_checked and self.identifier is not None:
            with self.stats.stage("face"):
                f.face_box = self.identifier.largest_face_box(f.clean,
                                                              config.SUIT_FACE_DETECT_WIDTH)
            f.face_checked = True
        # the whole body becomes the suit; only the pen stays visible
        now_s = time.monotonic()
        source = (self.cap.index, self.mode_session)
        with self.stats.stage("pose"):
            # contour searched once; pose and suit reuse it (#57)
            body = pose_tracking.find_player(mask)
            body_pose = self.pose_tracker.update(f.clean, mask, now=now_s, source=source,
                                                 captured_at=f.now, player=body)
            if self.pose_tracker._last_verified is not None:
                self.stats.maximum("pose_age_ms",
                                   max(0.0, now_s - self.pose_tracker._last_verified) * 1000.0)
            cost = getattr(self.pose_tracker.infer, "last_inference_seconds", None)
            if cost is not None:
                self.stats.add("pose_infer", cost)
        with self.stats.stage("suit"):
            f.frame = suit.dress(f.frame, mask, f.face_box, f.pen_mask, now_s,
                                 original=f.clean, pose=body_pose,
                                 renderer=self.suit_renderer, source=source, player=body)
        # the saber hilt attaches to these gloves (pose socket, no extra inference)
        f.saber_hands = lightsaber.glove_grips(self.suit_renderer.last_parts)

    # -- screens ---------------------------------------------------------------------------
    def _run_screen(self, f):
        t_stage = time.perf_counter()
        name = self.SCREENS.get(self.state)
        if name is not None:
            getattr(self, name)(f)
        self.stats.add("screen", time.perf_counter() - t_stage)

    def _menu_screen(self, f):
        w, h = config.WIDTH, config.HEIGHT
        start_t, quit_t = menu_start_target(w, h), quit_target(w, h)
        sbtn, swatches = settings_button(w, h), color_swatches(w, h)
        cbtn = camera_button(w, h)
        self.previews[self.cap.index] = cv2.resize(f.clean, cameras.PREVIEW_SIZE)
        available = sorted(self.previews)
        cam_tiles = camera_tiles(w, h, len(available))
        self._menu_clicks(available, sbtn, swatches)

        player_t = player_target(w, h) if self.identifier else None
        challenge_t = challenge_target(w, h)
        targets = menu_targets(w, h, self.identifier is not None)
        if self.settings_open or self.camera_open:
            active, progress, confirmed = None, 0.0, None
        else:
            active, progress, confirmed = self.dwell.update(f.point, targets)
        draw_menu(f.frame, start_t, quit_t, sbtn, swatches, self.pen_color,
                  self.settings_open, active, progress, f.point, player_t,
                  self.player if self.identifier else None,
                  blocked=[c for c in COLORS if not pen_allowed(c, self.bg_mode)],
                  challenge_t=challenge_t)
        if self.hint_text and time.time() < self.hint_until:
            draw_text(f.frame, self.hint_text, (config.WIDTH // 2, int(config.HEIGHT * 0.68)),
                      scale=0.6, color=(80, 80, 255), center=True)
        if not self.settings_open and not self.camera_open:
            _draw_background_button(f.frame, background_button(w, h), self.bg_mode)
        if show_camera_button(available):
            _draw_camera_button(f.frame, cbtn, self.cap.index, self.camera_open)
        if self.camera_open:
            draw_camera_panel(f.frame, cam_tiles, available, self.cap.index, self.previews)
        self._menu_confirmed(f, confirmed)

    def _menu_clicks(self, available, sbtn, swatches):
        """A mouse click in the menu: camera selection, background mode or pen colour."""
        w, h = config.WIDTH, config.HEIGHT
        if self.click is not None:
            was_open = self.camera_open
            self.camera_open = handle_camera_click(self.click, self.camera_open, available,
                                                   self.cap, w, h)
            if was_open or self.camera_open:  # click belonged to the camera selection
                self.settings_open = False
                self.dwell.reset()
                self.click = None
        if self.click is not None:
            new_mode = handle_background_click(self.click, self.bg_mode, w, h)
            if new_mode != self.bg_mode:          # click cycled the background mode
                self._set_background_mode(new_mode)
                self.settings_open = False
                self.dwell.reset()
                self.click = None
        if self.click is not None:
            if _in_rect(self.click, sbtn):
                self.settings_open = not self.settings_open
                self.dwell.reset()
            elif self.settings_open:
                self._pick_swatch(swatches)
                self.settings_open = False        # any click in the panel closes it
            self.click = None

    def _set_background_mode(self, new_mode):
        self.bg_mode = new_mode
        self.mode_session += 1
        if self.bg_mode == background.MODE_YOLO and self.yolo is not None:
            self.yolo.retry()                     # chosen again: a failed load may retry
        if not pen_allowed(self.pen_color, self.bg_mode):
            self.pen_color = pen_for_mode(self.pen_color, self.bg_mode)
            self.tracker.set_color(self.pen_color)
            logger.info("Pen colour switched to %s (not usable with %s)",
                        self.pen_color.name, self.bg_mode)
            self.hint_text, self.hint_until = PEN_HINT, time.time() + 3.0
        cameras.save_background_mode(config.SETTINGS_FILE, self.bg_mode)
        logger.info("Background mode set to %s (mouse)", self.bg_mode)

    def _pick_swatch(self, swatches):
        for i, sw_t in enumerate(swatches):
            if _hit(self.click, sw_t):
                if not pen_allowed(COLORS[i], self.bg_mode):
                    logger.info("Pen colour %s refused in %s mode", COLORS[i].name, self.bg_mode)
                    self.hint_text, self.hint_until = PEN_HINT, time.time() + 3.0
                    break
                self.pen_color = COLORS[i]
                self.tracker.set_color(self.pen_color)
                logger.info("Pen colour set to %s (mouse)", self.pen_color.name)
                break

    def _menu_confirmed(self, f, confirmed):
        if confirmed == MENU_START:
            self.play_mode = MODE_CLASSIC
            logger.info("Start (colour %s, player %s)", self.pen_color.name, self.player)
            self.dwell.reset(require_exit=True)
            self.state = STATE_DIFFICULTY
        elif confirmed == MENU_CHALLENGE:
            self.play_mode = MODE_CHALLENGE
            logger.info("Start challenge (colour %s, player %s)", self.pen_color.name,
                        self.player)
            self.dwell.reset(require_exit=True)
            self.state = STATE_DIFFICULTY
        elif confirmed == MENU_QUIT:
            f.quit_requested = True
        elif confirmed == MENU_PLAYER:            # Spieler wechseln
            logger.info("Switch player")
            self.dwell.reset(require_exit=True)
            self.scanner.reset()
            self.greet_until = None
            self.state = STATE_FACE_SCAN

    def _face_scan_screen(self, f):
        w, h = config.WIDTH, config.HEIGHT
        circle = face_circle(w, h)
        guest_t = guest_target(w, h)
        # camera selection here too: with the wrong camera nobody could be scanned
        self.previews[self.cap.index] = cv2.resize(f.clean, cameras.PREVIEW_SIZE)
        available = sorted(self.previews)
        if self.click is not None:
            self.camera_open = handle_camera_click(self.click, self.camera_open, available,
                                                   self.cap, w, h)
            self.click = None
            self.scanner.reset()                  # restart the scan with the (new) camera
        name, score, box, crop = f.scan_result    # detected once at the top of the frame
        inside = face_id.face_in_circle(box, circle) and not self.camera_open
        g_active, g_progress, g_confirmed = self.dwell.update(f.point, [guest_t])
        if self.camera_open:
            g_confirmed = None
        if self.greet_until is None:
            self._face_scan_result(inside, name, crop, g_confirmed)
        if self.state == STATE_FACE_SCAN:
            draw_face_scan(f.frame, circle, box, inside, self.scanner.progress,
                           self.player if self.greet_until is not None else None, guest_t,
                           g_active == 0, g_progress, f.point)
            if show_camera_button(available):
                _draw_camera_button(f.frame, camera_button(w, h), self.cap.index,
                                    self.camera_open)
            if self.camera_open:
                draw_camera_panel(f.frame, camera_tiles(w, h, len(available)), available,
                                  self.cap.index, self.previews)
            if self.greet_until is not None and time.time() >= self.greet_until:
                self.dwell.reset(require_exit=True)
                self.state = STATE_MENU

    def _face_scan_result(self, inside, name, crop, g_confirmed):
        result = self.scanner.update(inside, name, crop)
        if g_confirmed is not None:
            result = ("known", face_id.GUEST)
        if result is not None and result[0] == "known":
            self.player = result[1]
            self.greet_until = time.time() + config.FACE_GREETING_SECONDS
            logger.info("Player recognised: %s", self.player)
        elif result is not None:
            self.new_crops = result[1]
            self.entry = keyboard.NameEntry()
            self.name_message, self.name_message_until = None, 0.0
            self.key_dwell.reset(require_exit=True)
            self.typed_key = None
            logger.info("New face recorded (%d crops) -- asking for the name",
                        len(self.new_crops))
            self.state = STATE_NAME_ENTRY

    def _name_entry_screen(self, f):
        k_targets = keyboard.layout(config.WIDTH, config.HEIGHT)
        active, progress, confirmed = self.key_dwell.update(f.point, k_targets)
        pressed = self.typed_key
        self.typed_key = None
        if confirmed is not None:
            pressed = keyboard.KEYS[confirmed]
            self.key_dwell.reset()                # hold again to repeat the same key
        name_confirmed = pressed is not None and self.entry.press(pressed)
        if name_confirmed:
            problem = face_id.name_problem(self.entry.result(), self.identifier.persons)
            if problem:                           # "Gast" or taken: stay here and say why (#70)
                logger.info("Name %r refused: %s", self.entry.result(), problem)
                self.name_message, self.name_message_until = problem, time.time() + 3.0
                name_confirmed = False
        if name_confirmed:
            self._enroll_player()
        else:
            draw_name_entry(f.frame, k_targets, self.entry, active, progress, f.point,
                            self.pen_color.display,
                            self.name_message if time.time() < self.name_message_until else None)

    def _enroll_player(self):
        self.player = self.entry.result() or face_id.default_player_name(self.identifier.persons)
        enrolled = self.identifier.enroll(self.faces_dir, self.player, self.new_crops)
        if enrolled.saved < enrolled.added:       # e.g. folder not writable (#65)
            self.hint_text, self.hint_until = NOT_SAVED_HINT, time.time() + 4.0
        self.new_crops = []
        self.scanner.reset()
        self.greet_until = time.time() + config.FACE_GREETING_SECONDS
        self.state = STATE_FACE_SCAN              # shows the greeting, then the menu

    def _difficulty_screen(self, f):
        d_targets = row_targets(config.WIDTH, config.HEIGHT, len(LEVELS))
        q_target = quit_target(config.WIDTH, config.HEIGHT)
        active, progress, confirmed = self.dwell.update(f.point, d_targets + [q_target])
        draw_difficulty_screen(f.frame, d_targets, q_target, active, progress, f.point,
                               self.pen_color.display)
        if confirmed is None:
            return
        if confirmed == len(d_targets):
            f.quit_requested = True
            return
        self.difficulty = LEVELS[confirmed]
        self.game = new_game(self.play_mode, self.difficulty, self.sounds)
        self.sounds.play("combo")
        logger.info("Difficulty selected: %s (%s)", self.difficulty.name, self.play_mode)
        self.blade.reset()
        self.motion.reset()
        self.dwell.reset(require_exit=True)
        self.state = STATE_PLAY

    def _play_screen(self, f):
        game = self.game
        self.blade.add(f.point)
        self.motion.add(f.point, f.now)           # f.now: time right after the frame was read
        game.update(self.blade, f.dt, self.motion)    # seconds, not frames (#42, PERF-01)
        game.draw(f.frame)
        self.blade.draw(f.frame, config.SABER_COLOR if config.SABER_ENABLED else None)
        game.draw_hud(f.frame)
        game.draw_combo(f.frame)
        game.draw_feedback(f.frame)
        if game.game_over:
            result = highscore.add_entry(config.HIGHSCORE_FILE, game.board_name,
                                         game.score, self.player)
            self.scores, self.rank, self.score_saved = result.board, result.rank, result.saved
            self.sounds.play("over")
            logger.info("Game over -- %s, %s scored %d (rank: %s)",
                        game.board_name, self.player, game.score, self.rank)
            self.dwell.reset(require_exit=True)
            self.state = STATE_GAME_OVER

    def _game_over_screen(self, f):
        game = self.game
        targets = row_targets(config.WIDTH, config.HEIGHT, 3, y_frac=0.80, r=50)
        active, progress, confirmed = self.dwell.update(f.point, targets)
        game.draw(f.frame)
        game.draw_hud(f.frame)
        draw_game_over(f.frame, game, self.scores, self.rank, targets, active, progress,
                       f.point, self.pen_color.display, self.player, self.score_saved)
        if confirmed == 0:                        # Neustart
            logger.info("Restart requested (%s)", game.board_name)
            game.reset()
            self.blade.reset()
            self.motion.reset()
            self.rank = None
            self.dwell.reset(require_exit=True)
            self.state = STATE_PLAY
        elif confirmed == 1:                      # Startmenue
            logger.info("Back to start menu")
            self.rank = None
            self.dwell.reset(require_exit=True)
            self.state = STATE_MENU
        elif confirmed == 2:                      # Quit
            f.quit_requested = True

    # -- on top of every screen -------------------------------------------------------------
    def _draw_status_text(self, f):
        w, h = config.WIDTH, config.HEIGHT
        if self.bg_capture_at is not None:
            secs = max(0, int(self.bg_capture_at - time.monotonic()) + 1)
            draw_text(f.frame, f"Aus dem Bild gehen: {secs}", (w // 2, int(h * 0.12)),
                      scale=1.0, center=True)
        elif (config.BG_ENABLED and self.space_img is not None and self.reference is None
              and self.bg_mode == background.MODE_NORMAL    # greenscreen needs no reference
              and self.state != STATE_NAME_ENTRY):          # there 'b' types a letter
            draw_text(f.frame, "Taste b: Weltall-Hintergrund", (w // 2, int(h * 0.96)),
                      scale=0.5, center=True)
        elif self.bg_mode == background.MODE_YOLO and (self.yolo is None
                                                       or self.yolo.get() is None):
            loading = self.yolo is not None and self.yolo.state in (self.yolo.IDLE,
                                                                    self.yolo.LOADING)
            draw_text(f.frame, "YOLO-Modell wird geladen ..." if loading
                      else "YOLO-Modell nicht verfuegbar", (w // 2, int(h * 0.96)), scale=0.5,
                      color=(200, 200, 200) if loading else (80, 80, 255), center=True)

    def _draw_debug_mask(self):
        if self.show_mask and self.tracker.last_mask is not None:
            cv2.imshow("mask (debug)", cv2.cvtColor(self.tracker.last_mask, cv2.COLOR_GRAY2BGR))

    def _draw_saber(self, f):
        """The pen as a lightsaber, on every screen (drawn last, like a cursor)."""
        if not config.SABER_ENABLED:
            return
        # tip = pen point (the only thing that cuts); the hilt sits on the glove the
        # suit drew from the pose (#46), otherwise on the hand found by skin colour --
        # the skin mask is then computed only when needed; the face is never the hand
        if not f.face_checked and self.identifier is not None and f.point is not None:
            with self.stats.stage("face"):
                f.face_box = self.identifier.largest_face_box(f.clean,
                                                              config.SUIT_FACE_DETECT_WIDTH)
            f.face_checked = True
        with self.stats.stage("saber"):
            cached_skin=[]
            def skin():
                if not cached_skin:cached_skin.append(lightsaber.skin_mask(f.clean))
                return cached_skin[0]
            source=(self.cap.index,self.mode_session,self.bg_mode,f.frame.shape,self.pen_color.name)
            weapon=self.saber.update(f.point,skin if f.point is not None else None,f.face_box,
                                     hands=f.saber_hands,now=f.now,source=source)
            occlusion=None
            if f.saber_hands and weapon is not None:
                occlusion=lightsaber.glove_occlusion(
                    self.suit_renderer.last_parts,self.saber.hand_side,
                    self.suit_renderer.assets.get('glove'),f.frame.shape,weapon)
            if occlusion is None and cached_skin:
                occlusion=(0,0,cached_skin[0])
            lightsaber.draw_saber(f.frame,weapon,config.SABER_COLOR,
                                  occlusion=occlusion,opacity=self.saber.visibility)

    def _show(self, f):
        """Show the picture and return the pressed key (the stage timer closes the frame)."""
        if self.show_perf:
            draw_perf_overlay(f.frame, self.stats.overlay_lines())
        with self.stats.stage("show"):
            cv2.imshow(config.WINDOW_NAME, f.frame)
            key = cv2.waitKey(1) & 0xFF
        self.stats.end_frame()
        return key

    def _handle_key(self, f, key):
        """Returns True when the game should end."""
        if self.state == STATE_NAME_ENTRY and keyboard.from_keycode(key) is not None:
            self.typed_key = keyboard.from_keycode(key)   # letters type here instead of q/d
        elif f.quit_requested or key == ord("q"):
            logger.info("Quit requested")
            return True
        elif key == ord("p"):
            self.show_perf = not self.show_perf
            logger.info("Perf overlay %s", "on" if self.show_perf else "off")
        elif key == ord("d"):
            self.show_mask = not self.show_mask
            logger.info("Debug mask %s", "on" if self.show_mask else "off")
            if not self.show_mask:
                cv2.destroyWindow("mask (debug)")
        elif key == ord("b") and self.space_img is not None:
            self.bg_capture_at = time.monotonic() + config.BG_COUNTDOWN_SECONDS
            logger.info("Background (re)capture armed -- %.1fs countdown",
                        config.BG_COUNTDOWN_SECONDS)
        return False


def _run_game_loop(cap, sounds, previews=None, identifier=None):
    if config.ASYNC_INFERENCE:
        cv2.setNumThreads(max(1, config.INFERENCE_CV_THREADS))
    with LatestWorker() if config.ASYNC_INFERENCE else nullcontext() as worker:
        return _game_loop(cap, sounds, previews, identifier, worker)


def _game_loop(cap, sounds, previews=None, identifier=None, worker=None):
    GameLoop(cap, sounds, previews, identifier, worker).run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suit-diagnostics', action='store_true',
                        help='Log aggregate live processing times every five seconds')
    arguments = parser.parse_args()
    config.SUIT_DIAGNOSTICS = config.SUIT_DIAGNOSTICS or arguments.suit_diagnostics
    main()
