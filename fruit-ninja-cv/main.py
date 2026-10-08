"""Fruit Ninja CV -- entry point and game loop.

States:
  MENU       Choose a difficulty (1 Einfach / 2 Mittel / 3 Schwer).
  CALIBRATE  Click the pink pen (best), or hold it in the box + SPACE; ENTER
             starts with the default pink range.
  PLAY       Slice thrown fruits with the pink pen (touch is enough), avoid bombs.
  GAME_OVER  Shows the final score and the Top-3 highscores.

Keys: q quit | 1/2/3 difficulty | r restart | m menu | c re-calibrate | d mask
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
from sound import SoundPlayer
import highscore

logger = logging.getLogger(__name__)

STATE_MENU = "menu"
STATE_CALIBRATE = "calibrate"
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


def _dim(frame, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_menu(frame):
    h, w = frame.shape[:2]
    _dim(frame)
    draw_text(frame, "FRUIT NINJA CV", (w // 2, int(h * 0.22)), scale=1.6,
              color=(180, 0, 255), thickness=3, center=True)
    draw_text(frame, "Schwierigkeit waehlen:", (w // 2, int(h * 0.40)),
              scale=0.9, center=True)
    for i, lvl in enumerate(LEVELS):
        draw_text(frame, f"{i + 1} - {lvl.name}  ({lvl.lives} Leben)",
                  (w // 2, int(h * 0.52) + i * 36), scale=0.85, center=True)
    draw_text(frame, "q - Quit", (w // 2, int(h * 0.86)), scale=0.7, center=True)


def draw_mask_inset(frame, mask):
    """Show a small live preview of the detection mask in the top-right corner."""
    iw, ih = 192, 108
    small = cv2.cvtColor(cv2.resize(mask, (iw, ih)), cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    x0, y0 = w - iw - 10, 10
    frame[y0:y0 + ih, x0:x0 + iw] = small
    cv2.rectangle(frame, (x0, y0), (x0 + iw, y0 + ih), (180, 0, 255), 1)
    draw_text(frame, "Maske", (x0 + 4, y0 + ih - 8), scale=0.5)


def draw_calibration_overlay(frame, tracker, point):
    h, w = frame.shape[:2]
    x, y, bw, bh = tracker.calib_box_px(w, h)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (180, 0, 255), 2)
    top = int(h * 0.12)
    draw_text(frame, "Auf den pinken Stift KLICKEN zum Kalibrieren",
              (w // 2, top), scale=0.8, center=True)
    draw_text(frame, "oder Stift in die Box + SPACE", (w // 2, top + 34),
              scale=0.7, center=True)
    draw_text(frame, "ENTER = Standard-Pink   |   q = Quit",
              (w // 2, top + 64), scale=0.65, center=True)
    if point is not None:
        cv2.circle(frame, point, 12, (0, 255, 0), 2)
        draw_text(frame, "erkannt", (point[0] + 14, point[1]),
                  scale=0.6, color=(0, 255, 0))
    else:
        draw_text(frame, "nichts erkannt - klick auf den Stift",
                  (w // 2, int(h * 0.9)), scale=0.65,
                  color=(80, 80, 255), center=True)


def draw_game_over_overlay(frame, game, highscores, rank):
    h, w = frame.shape[:2]
    _dim(frame)
    draw_text(frame, "GAME OVER", (w // 2, int(h * 0.22)), scale=1.8,
              color=(80, 80, 255), thickness=3, center=True)
    draw_text(frame, f"Score: {game.score}", (w // 2, int(h * 0.33)),
              scale=1.1, center=True)
    if rank is not None:
        draw_text(frame, f"NEUER HIGHSCORE  (#{rank})!", (w // 2, int(h * 0.41)),
                  scale=0.8, color=(0, 215, 255), center=True)

    draw_text(frame, "Top 3", (w // 2, int(h * 0.52)), scale=0.8, center=True)
    for i in range(3):
        value = highscores[i] if i < len(highscores) else "-"
        draw_text(frame, f"{i + 1}.  {value}", (w // 2, int(h * 0.58) + i * 32),
                  scale=0.75, center=True)

    draw_text(frame, "r = Neustart    m = Menue    q = Quit",
              (w // 2, int(h * 0.88)), scale=0.75, center=True)


def main():
    log_path = setup_logging()
    logger.info("Starting Fruit Ninja CV (logfile: %s)", log_path)

    cap = None
    try:
        cap = open_camera()
        sounds = SoundPlayer()
        highscores = highscore.load(config.HIGHSCORE_FILE)
        _run_game_loop(cap, sounds, highscores)
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


def _run_game_loop(cap, sounds, highscores):
    tracker = MarkerTracker()
    blade = Blade()
    game = None
    difficulty = None
    rank = None                 # highscore rank of the last finished run
    state = STATE_MENU
    show_mask = False
    click = [None]              # mutable holder so the mouse callback can post a click

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click[0] = (x, y)

    cv2.namedWindow(config.WINDOW_NAME)
    cv2.setMouseCallback(config.WINDOW_NAME, on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.error("Lost camera frame -- aborting game loop")
            print("ERROR: Lost camera frame.", file=sys.stderr)
            break

        frame = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
        if config.MIRROR:
            frame = cv2.flip(frame, 1)
        clean = frame.copy()    # overlay-free copy used for colour sampling

        # A click on the pen while calibrating is the most precise calibration
        if click[0] is not None:
            if state == STATE_CALIBRATE:
                hue = tracker.calibrate_from_point(clean, click[0][0], click[0][1])
                logger.info("Calibrated by click at %s -> hue %.0f", click[0], hue)
                blade.reset()
                state = STATE_PLAY
            click[0] = None

        if state == STATE_MENU:
            draw_menu(frame)

        elif state == STATE_CALIBRATE:
            point = tracker.track(clean)          # live detection for feedback
            draw_calibration_overlay(frame, tracker, point)
            if tracker.last_mask is not None:
                draw_mask_inset(frame, tracker.last_mask)

        elif state == STATE_PLAY:
            point = tracker.track(clean)
            blade.add(point)
            game.update(blade)
            game.draw(frame)
            blade.draw(frame)
            game.draw_hud(frame)
            game.draw_combo(frame)
            draw_text(frame, "c = neu kalibrieren", (16, config.HEIGHT - 16),
                      scale=0.6)
            if game.game_over:
                highscores, rank = highscore.add(config.HIGHSCORE_FILE, game.score)
                sounds.play("over")
                logger.info("Game over -- final score %d (highscore rank: %s)",
                            game.score, rank)
                state = STATE_GAME_OVER

        elif state == STATE_GAME_OVER:
            game.draw(frame)
            game.draw_hud(frame)
            draw_game_over_overlay(frame, game, highscores, rank)

        if show_mask and tracker.last_mask is not None:
            mask_bgr = cv2.cvtColor(tracker.last_mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow("mask (debug)", mask_bgr)

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
        elif state == STATE_MENU and key in (ord("1"), ord("2"), ord("3")):
            difficulty = LEVELS[key - ord("1")]
            game = Game(config.WIDTH, config.HEIGHT, difficulty, sounds)
            blade.reset()
            logger.info("Difficulty selected: %s", difficulty.name)
            state = STATE_CALIBRATE
        elif key == ord("c") and game is not None:
            logger.info("Re-calibration requested")
            state = STATE_CALIBRATE
            blade.reset()
        elif key == ord(" ") and state == STATE_CALIBRATE:
            hue = tracker.calibrate(frame)   # sample the pink pen from the box
            logger.info("Calibrated from box -> hue %.0f", hue)
            blade.reset()
            state = STATE_PLAY
        elif key in (13, 10) and state == STATE_CALIBRATE:
            logger.info("Starting with default pink range")
            blade.reset()                    # start with the default pink range
            state = STATE_PLAY
        elif key == ord("r") and state == STATE_GAME_OVER:
            logger.info("Restart requested (%s)", game.difficulty.name)
            game.reset()
            blade.reset()
            rank = None
            state = STATE_PLAY
        elif key == ord("m") and state == STATE_GAME_OVER:
            logger.info("Back to menu")
            rank = None
            state = STATE_MENU


if __name__ == "__main__":
    main()
