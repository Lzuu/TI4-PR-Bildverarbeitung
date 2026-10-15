"""Fruit Ninja CV -- entry point and game loop.

The window opens in fullscreen and the whole layout + physics scale to the
screen resolution (responsive). The app starts in a menu with the PINK pen
active by default (only pink is tracked). You start by holding the pen on the
"Start" circle (VR-style dwell). The pen colour is a *setting*: it is hidden
behind a "Stiftfarbe" button and only shown when you open it (mouse click).

States:
  MENU       Start / Quit (pen dwell); pen colour setting via mouse button.
  DIFFICULTY Pick the difficulty (Einfach / Mittel / Schwer) by dwelling.
  PLAY       Slice thrown fruits with the pen (touch is enough), avoid bombs.
  GAME_OVER  Score + Top-3 leaderboard; dwell on Neustart / Startmenue / Quit.

Keys: q quit | d mask (debug)
"""

import logging
import re
import subprocess
import sys

import cv2

import config
from marker_tracker import MarkerTracker
from blade import Blade
from game import Game
from utils import draw_text
from logging_config import setup_logging
from difficulty import LEVELS
from colors import COLORS
from dwell import DwellTracker
from sound import SoundPlayer
import highscore

logger = logging.getLogger(__name__)

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
    """Open the configured camera, falling back to index 0."""
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if cap.isOpened():
        logger.info("Opened camera at index %d", config.CAMERA_INDEX)
    elif config.CAMERA_INDEX != 0:
        logger.warning("Camera index %d unavailable, falling back to 0",
                       config.CAMERA_INDEX)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            logger.info("Opened fallback camera at index 0")
    if not cap.isOpened():
        logger.error("Could not open any camera (tried %d and 0)",
                     config.CAMERA_INDEX)
        print("ERROR: Could not open any camera.", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


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
              active, progress, point):
    _dim(frame)
    h, w = frame.shape[:2]
    draw_text(frame, "FRUIT NINJA CV", (w // 2, int(h * 0.16)), scale=1.4,
              color=(180, 0, 255), thickness=3, center=True)
    _draw_circle_option(frame, start_t, pen_color.display, "Start", active == 0,
                        progress)
    _draw_circle_option(frame, quit_t, (200, 200, 200), "Quit", active == 1,
                        progress)
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
                   color):
    _dim(frame, 0.55)
    h, w = frame.shape[:2]
    draw_text(frame, "GAME OVER", (w // 2, int(h * 0.16)), scale=1.6,
              color=(80, 80, 255), thickness=3, center=True)
    draw_text(frame, f"Score: {game.score}", (w // 2, int(h * 0.26)),
              scale=1.0, center=True)
    if rank is not None:
        draw_text(frame, f"NEUER HIGHSCORE  (#{rank})!", (w // 2, int(h * 0.33)),
                  scale=0.75, color=(0, 215, 255), center=True)
    draw_text(frame, f"Leaderboard - {game.difficulty.name}",
              (w // 2, int(h * 0.42)), scale=0.7, center=True)
    for i in range(3):
        value = scores[i] if i < len(scores) else "-"
        draw_text(frame, f"{i + 1}.  {value}", (w // 2, int(h * 0.47) + _s(26) * i),
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
        cap = open_camera()
        sounds = SoundPlayer()
        _run_game_loop(cap, sounds)
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


def _run_game_loop(cap, sounds):
    tracker = MarkerTracker()
    blade = Blade()
    dwell = DwellTracker(config.DWELL_SECONDS)
    game = None
    difficulty = None
    pen_color = COLORS[0]            # Pink by default -- only this colour is tracked
    tracker.set_color(pen_color)
    scores, rank = [], None
    show_mask = False
    settings_open = False
    frame_failures = 0
    click = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click[0] = (x, y)

    state = STATE_MENU
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

        quit_requested = False

        if state == STATE_MENU:
            start_t = menu_start_target(config.WIDTH, config.HEIGHT)
            quit_t = quit_target(config.WIDTH, config.HEIGHT)
            sbtn = settings_button(config.WIDTH, config.HEIGHT)
            swatches = color_swatches(config.WIDTH, config.HEIGHT)

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
            if settings_open:
                active, progress, confirmed = None, 0.0, None
            else:
                active, progress, confirmed = dwell.update(point, [start_t, quit_t])
            draw_menu(frame, start_t, quit_t, sbtn, swatches, pen_color,
                      settings_open, active, progress, point)
            if confirmed == 0:              # Start
                logger.info("Start (colour %s)", pen_color.name)
                dwell.reset()
                state = STATE_DIFFICULTY
            elif confirmed == 1:            # Quit
                quit_requested = True

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
                                             game.difficulty.name, game.score)
                sounds.play("over")
                logger.info("Game over -- %s, score %d (rank: %s)",
                            game.difficulty.name, game.score, rank)
                dwell.reset()
                state = STATE_GAME_OVER

        elif state == STATE_GAME_OVER:
            point = tracker.track(clean)
            targets = row_targets(config.WIDTH, config.HEIGHT, 3, y_frac=0.80, r=50)
            active, progress, confirmed = dwell.update(point, targets)
            game.draw(frame)
            game.draw_hud(frame)
            draw_game_over(frame, game, scores, rank, targets, active, progress,
                           point, pen_color.display)
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

        if show_mask and tracker.last_mask is not None:
            cv2.imshow("mask (debug)",
                       cv2.cvtColor(tracker.last_mask, cv2.COLOR_GRAY2BGR))

        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if quit_requested or key == ord("q"):
            logger.info("Quit requested")
            break
        elif key == ord("d"):
            show_mask = not show_mask
            logger.info("Debug mask %s", "on" if show_mask else "off")
            if not show_mask:
                cv2.destroyWindow("mask (debug)")


if __name__ == "__main__":
    main()
