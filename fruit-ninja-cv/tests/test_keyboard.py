"""Tests for the on-screen keyboard used to type a new player's name with the pen."""

import config
import keyboard


def type_keys(entry, keys):
    done = False
    for k in keys:
        done = entry.press(k)
    return done


def test_one_target_per_key_inside_the_screen_without_overlap():
    w, h = config.WIDTH, config.HEIGHT
    targets = keyboard.layout(w, h)
    assert len(targets) == len(keyboard.KEYS)
    for i, (cx, cy, r) in enumerate(targets):
        assert r <= cx <= w - r and r <= cy <= h - r
        for ox, oy, orr in targets[i + 1:]:
            assert (cx - ox) ** 2 + (cy - oy) ** 2 >= (r + orr) ** 2


def test_keyboard_has_letters_umlauts_and_controls():
    for k in ["A", "Z", "Ä", "Ö", "Ü", keyboard.SPACE, keyboard.DELETE, keyboard.DONE]:
        assert k in keyboard.KEYS


def test_words_are_capitalised_automatically():
    entry = keyboard.NameEntry()
    type_keys(entry, ["N", "O", "A", "H", keyboard.SPACE, "B", "R", "A", "U", "N"])
    assert entry.text == "Noah Braun"


def test_delete_removes_the_last_character():
    entry = keyboard.NameEntry()
    type_keys(entry, ["L", "E", "O", keyboard.DELETE])
    assert entry.text == "Le"
    type_keys(entry, [keyboard.DELETE] * 5)          # deleting past the start is harmless
    assert entry.text == ""


def test_no_leading_or_double_spaces():
    entry = keyboard.NameEntry()
    type_keys(entry, [keyboard.SPACE, "A", keyboard.SPACE, keyboard.SPACE, "B"])
    assert entry.text == "A B"


def test_name_length_is_limited():
    entry = keyboard.NameEntry(max_length=5)
    type_keys(entry, list("ABCDEFGH"))
    assert entry.text == "Abcde"


def test_done_finishes_and_strips_trailing_space():
    entry = keyboard.NameEntry()
    assert not type_keys(entry, ["L", "E", "O", keyboard.SPACE])
    assert entry.press(keyboard.DONE)
    assert entry.result() == "Leo"


def test_physical_keyboard_maps_to_the_same_keys():
    assert keyboard.from_keycode(ord("a")) == "A"
    assert keyboard.from_keycode(ord(" ")) == keyboard.SPACE
    assert keyboard.from_keycode(8) == keyboard.DELETE        # Backspace
    assert keyboard.from_keycode(13) == keyboard.DONE         # Enter
    assert keyboard.from_keycode(255) is None                 # no key pressed
