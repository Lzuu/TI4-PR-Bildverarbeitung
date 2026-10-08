"""Fruit Ninja CV -- entry point and game loop.

The app opens in a start menu with the PINK pen active by default (only pink is
tracked -- no false positives). You start the game by holding the pen on the
"Start" circle (VR-style dwell). The pen colour is only changed if you want to:
click one of the colour swatches with the MOUSE. Everything else in-game is
pen-driven.

States:
  MENU       Start (pen dwell) + Quit (pen dwell); pen colour via MOUSE click.
  DIFFICULTY Pick the difficulty (Einfach / Mittel / Schwer) by dwelling.
  PLAY       Slice thrown fruits with the pen (touch is enough), avoid bombs.
  GAME_OVER  Score + Top-3 leaderboard; dwell on Neustart / Startmenue / Quit.

Keys: q quit | d mask (debug)
"""

import logging
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)
    return cap


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def row_targets(w, h, n, y_frac=0.46, r=config.SELECT_RADIUS):
    """n evenly spread circles in a horizontal row: list of (cx, cy, r)."""
    y = int(h * y_frac)
    return [(int(w * (i + 1) / (n + 1)), y, r) for i in range(n)]


def quit_target(w, h):
    return (int(w * 0.5), int(h * 0.86), 40)


def menu_start_target(w, h):
    return (int(w * 0.5), int(h * 0.38), config.CONFIRM_RADIUS)


def color_swatches(w, h):
    y = int(h * 0.64)
    r = 26
    xs = [int(w * 0.42), int(w * 0.5), int(w * 0.58)]
    return [(xs[i], y, r) for i in range(len(COLORS))]


def _hit(point, target):
    px, py = point
    cx, cy, r = target
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def _dim(frame, alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_ring(frame, center, r, progress, color):
    if progress > 0:
        cv2.ellipse(frame, center, (r, r), -90, 0, int(360 * progress), color, 8,
                    cv2.LINE_AA)


def _draw_circle_option(frame, target, color, label, active, progress):
    cx, cy, r = target
    cv2.circle(frame, (cx, cy), r, color, 3, cv2.LINE_AA)
    if active:
        _draw_ring(frame, (cx, cy), r, progress, color)
    draw_text(frame, label, (cx, cy + r + 26), scale=0.75, color=color, center=True)


def _draw_pen_marker(frame, point):
    if point is not None:
        cv2.circle(frame, point, 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
    return point is not None


def draw_menu(frame, start_t, quit_t, swatches, pen_color, active, progress, point):
    _dim(frame)
    h, w = frame.shape[:2]
    draw_text(frame, "FRUIT NINJA CV", (w // 2, int(h * 0.16)), scale=1.4,
              color=(180, 0, 255), thickness=3, center=True)
    _draw_circle_option(frame, start_t, pen_color.display, "Start", active == 0,
                        progress)
    _draw_circle_option(frame, quit_t, (200, 200, 200), "Quit", active == 1,
                        progress)
    # Colour setting (mouse-clickable swatches)
    draw_text(frame, "Stiftfarbe (Maus-Klick):", (w // 2, int(h * 0.56)),
              scale=0.6, center=True)
    for i, (cx, cy, r) in enumerate(swatches):
        cv2.circle(frame, (cx, cy), r, COLORS[i].display, -1, cv2.LINE_AA)
        if COLORS[i] is pen_color:          # highlight the active colour
            cv2.circle(frame, (cx, cy), r + 5, (255, 255, 255), 2, cv2.LINE_AA)
    if not _draw_pen_marker(frame, point):
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
    iw, ih = 192, 108
    small = cv2.cvtColor(cv2.resize(mask, (iw, ih)), cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    x0, y0 = w - iw - 10, 10
    frame[y0:y0 + ih, x0:x0 + iw] = small
    cv2.rectangle(frame, (x0, y0), (x0 + iw, y0 + ih), (255, 255, 255), 1)
    draw_text(frame, "Maske", (x0 + 4, y0 + ih - 8), scale=0.5)


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
        draw_text(frame, f"{i + 1}.  {value}", (w // 2, int(h * 0.47) + i * 26),
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
    frame_failures = 0
    click = [None]                   # mouse clicks (colour setting in the menu)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click[0] = (x, y)

    state = STATE_MENU
    cv2.namedWindow(config.WINDOW_NAME)
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

        frame = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
        if config.MIRROR:
            frame = cv2.flip(frame, 1)
        clean = frame.copy()

        quit_requested = False

        if state == STATE_MENU:
            start_t = menu_start_target(config.WIDTH, config.HEIGHT)
            quit_t = quit_target(config.WIDTH, config.HEIGHT)
            swatches = color_swatches(config.WIDTH, config.HEIGHT)
            # Mouse click changes the pen colour (the only mouse interaction)
            if click[0] is not None:
                for i, sw in enumerate(swatches):
                    if _hit(click[0], sw):
                        pen_color = COLORS[i]
                        tracker.set_color(pen_color)
                        dwell.reset()
                        logger.info("Pen colour set to %s (mouse)", pen_color.name)
                        break
                click[0] = None

            point = tracker.track(clean)
            active, progress, confirmed = dwell.update(point, [start_t, quit_t])
            draw_menu(frame, start_t, quit_t, swatches, pen_color, active,
                      progress, point)
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
            targets = row_targets(config.WIDTH, config.HEIGHT, 3, y_frac=0.80, r=46)
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
