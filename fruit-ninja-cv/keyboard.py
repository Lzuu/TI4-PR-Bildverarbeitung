"""On-screen keyboard: a new player types their name with the pen (dwell per key).

The keys are circles like the other dwell targets; the physical keyboard works
as a fallback (letters, space, Backspace, Enter).
"""

import config

SPACE = "Leer"
DELETE = "<-"
DONE = "OK"
KEYS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Ä", "Ö", "Ü", SPACE, DELETE, DONE]
COLUMNS = 8


def layout(w, h):
    """Dwell targets (cx, cy, r), one per key, as a centred grid in the lower part."""
    rows = -(-len(KEYS) // COLUMNS)
    top, bottom = h * 0.36, h * 0.97
    pitch = min(w * 0.9 / COLUMNS, (bottom - top) / rows)
    r = int(pitch * 0.42)
    x0 = (w - pitch * (COLUMNS - 1)) / 2
    y0 = top + pitch / 2
    return [(int(x0 + (i % COLUMNS) * pitch), int(y0 + (i // COLUMNS) * pitch), r)
            for i in range(len(KEYS))]


def from_keycode(code):
    """Map a cv2.waitKey code to a keyboard key (or None)."""
    if code in (8, 127):
        return DELETE
    if code in (10, 13):
        return DONE
    if code == 32:
        return SPACE
    # Only plain A-Z: 255 means "no key" in cv2.waitKey and chr(255) == "ÿ" is a letter
    if ord("a") <= code <= ord("z") or ord("A") <= code <= ord("Z"):
        return chr(code).upper()
    return None


class NameEntry:
    """The name being typed; words are capitalised automatically."""

    def __init__(self, max_length=config.NAME_MAX_LENGTH):
        self.max_length = max_length
        self.text = ""

    def press(self, key):
        """Apply a key. Returns True once the name is confirmed (DONE)."""
        if key == DONE:
            return True
        if key == DELETE:
            self.text = self.text[:-1]
        elif key == SPACE:
            if self.text and not self.text.endswith(" ") and len(self.text) < self.max_length:
                self.text += " "
        elif len(self.text) < self.max_length:
            new_word = not self.text or self.text.endswith(" ")
            self.text += key.upper() if new_word else key.lower()
        return False

    def result(self):
        return self.text.strip()
