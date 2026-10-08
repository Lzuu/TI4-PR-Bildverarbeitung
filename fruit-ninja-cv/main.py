"""Fruit Ninja CV -- entry point and game loop.

Selection is VR-style: hold the pen inside a circle for a few seconds and a
ring fills up to confirm -- no keyboard needed and no colour self-calibration.

States:
  COLOR      Pick the pen colour (Pink / Gelb / Gruen) by dwelling in a circle.
  DIFFICULTY Pick the difficulty (Einfach / Mittel / Schwer) by dwelling.
  PLAY       Slice thrown fruits with the pen (touch is enough), avoid bombs.
  GAME_OVER  Shows the final score and the Top-3 leaderboard for that difficulty.

Keys: q quit | r restart | m back to selection | d mask (debug)
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

STATE_COLOR = "color"
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
# Drawing helpers
# --------------------------------------------------------------------------- #
def _dim(frame, alpha=0.5):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def selection_targets(w, h, n=3):
    """Centres + radius of the n selection circles (evenly spread)."""
    y = int(h * 0.52)
    r = config.SELECT_RADIUS
    xs = [int(w * (i + 1) / (n + 1)) for i in range(n)]
    return [(x, y, r) for x in xs]


def _draw_ring(frame, center, r, progress, color):
    """A circular progress ring (fills clockwise from the top)."""
    if progress > 0:
        cv2.ellipse(frame, center, (r, r), -90, 0, int(360 * progress), color, 8,
                    cv2.LINE_AA)


def draw_selection(frame, title, labels, ring_colors, targets, active, progress,
                   point):
    _dim(frame)
    h, w = frame.shape[:2]
    draw_text(frame, title, (w // 2, int(h * 0.22)), scale=1.1, center=True)
    draw_text(frame, f"Stift {int(config.DWELL_SECONDS)}s in einen Kreis halten",
              (w // 2, int(h * 0.30)), scale=0.7, center=True)
    for i, (cx, cy, r) in enumerate(targets):
        col = ring_colors[i]
        cv2.circle(frame, (cx, cy), r, col, 3, cv2.LINE_AA)
        if active == i:
            _draw_ring(frame, (cx, cy), r, progress, col)
        draw_text(frame, labels[i], (cx, cy + r + 30), scale=0.8,
                  color=col, center=True)
    if point is not None:
        cv2.circle(frame, point, 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        draw_text(frame, "Stift nicht erkannt", (w // 2, int(h * 0.88)),
                  scale=0.65, color=(80, 80, 255), center=True)
    draw_text(frame, "q - Quit", (w // 2, int(h * 0.94)), scale=0.6, center=True)


def draw_mask_inset(frame, mask):
    iw, ih = 192, 108
    small = cv2.cvtColor(cv2.resize(mask, (iw, ih)), cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    x0, y0 = w - iw - 10, 10
    frame[y0:y0 + ih, x0:x0 + iw] = small
    cv2.rectangle(frame, (x0, y0), (x0 + iw, y0 + ih), (255, 255, 255), 1)
    draw_text(frame, "Maske", (x0 + 4, y0 + ih - 8), scale=0.5)


def draw_game_over(frame, game, scores, rank):
    _dim(frame, 0.55)
    h, w = frame.shape[:2]
    draw_text(frame, "GAME OVER", (w // 2, int(h * 0.20)), scale=1.8,
              color=(80, 80, 255), thickness=3, center=True)
    draw_text(frame, f"Score: {game.score}", (w // 2, int(h * 0.31)),
              scale=1.1, center=True)
    if rank is not None:
        draw_text(frame, f"NEUER HIGHSCORE  (#{rank})!", (w // 2, int(h * 0.39)),
                  scale=0.8, color=(0, 215, 255), center=True)
    draw_text(frame, f"Leaderboard - {game.difficulty.name}",
              (w // 2, int(h * 0.50)), scale=0.8, center=True)
    for i in range(3):
        value = scores[i] if i < len(scores) else "-"
        draw_text(frame, f"{i + 1}.  {value}", (w // 2, int(h * 0.56) + i * 30),
                  scale=0.72, center=True)
    draw_text(frame, "r = Neustart    m = Auswahl    q = Quit",
              (w // 2, int(h * 0.88)), scale=0.72, center=True)


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
    pen_color = None
    difficulty = None
    scores, rank = [], None
    show_mask = False

    state = STATE_COLOR
    tracker.set_colors(COLORS)          # colour screen: detect any pen colour

    cv2.namedWindow(config.WINDOW_NAME)

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.error("Lost camera frame -- aborting game loop")
            print("ERROR: Lost camera frame.", file=sys.stderr)
            break

        frame = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
        if config.MIRROR:
            frame = cv2.flip(frame, 1)
        clean = frame.copy()

        if state == STATE_COLOR:
            point = tracker.track(clean)
            targets = selection_targets(config.WIDTH, config.HEIGHT, len(COLORS))
            active, progress, confirmed = dwell.update(point, targets)
            draw_selection(frame, "Stiftfarbe waehlen",
                           [c.name for c in COLORS], [c.display for c in COLORS],
                           targets, active, progress, point)
            if confirmed is not None:
                pen_color = COLORS[confirmed]
                tracker.set_color(pen_color)
                sounds.play("combo")
                logger.info("Pen colour selected: %s", pen_color.name)
                dwell.reset()
                state = STATE_DIFFICULTY

        elif state == STATE_DIFFICULTY:
            point = tracker.track(clean)
            targets = selection_targets(config.WIDTH, config.HEIGHT, len(LEVELS))
            active, progress, confirmed = dwell.update(point, targets)
            ring = [pen_color.display] * len(LEVELS)
            draw_selection(frame, "Schwierigkeit waehlen",
                           [lvl.name for lvl in LEVELS], ring,
                           targets, active, progress, point)
            if confirmed is not None:
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
                state = STATE_GAME_OVER

        elif state == STATE_GAME_OVER:
            game.draw(frame)
            game.draw_hud(frame)
            draw_game_over(frame, game, scores, rank)

        if show_mask and tracker.last_mask is not None:
            cv2.imshow("mask (debug)",
                       cv2.cvtColor(tracker.last_mask, cv2.COLOR_GRAY2BGR))

        cv2.imshow(config.WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            logger.info("Quit requested by user")
            break
        elif key == ord("d"):
            show_mask = not show_mask
            logger.info("Debug mask %s", "on" if show_mask else "off")
            if not show_mask:
                cv2.destroyWindow("mask (debug)")
        elif key == ord("r") and state == STATE_GAME_OVER:
            logger.info("Restart requested (%s)", game.difficulty.name)
            game.reset()
            blade.reset()
            rank = None
            state = STATE_PLAY
        elif key == ord("m") and state == STATE_GAME_OVER:
            logger.info("Back to colour selection")
            tracker.set_colors(COLORS)
            dwell.reset()
            rank = None
            state = STATE_COLOR


if __name__ == "__main__":
    main()
