"""Teilaufgaben 3 und 4: Stiftverfolgung und Erkennung der Schnittgeste."""
import fruit_ninja as fn

DT = 1 / 30


def test_ohne_stift_wird_die_spur_geloescht():
    trail = [((100, 100), 0.0)]
    trail, blade = fn.update_blade(trail, None, DT)
    assert trail == []
    assert blade is None


def test_erste_position_ist_noch_kein_schnitt():
    trail, blade = fn.update_blade([], (100, 100), 0.0)
    assert trail == [((100, 100), 0.0)]
    assert blade is None


def test_schnelle_bewegung_ist_ein_schnitt():
    trail, _ = fn.update_blade([], (100, 100), 0.0)
    trail, blade = fn.update_blade(trail, (130, 100), DT)    # 900 px/s
    assert blade == ((100, 100), (130, 100))


def test_langsame_bewegung_ist_kein_schnitt():
    trail, _ = fn.update_blade([], (100, 100), 0.0)
    step = fn.MIN_SLICE_SPEED * DT * 0.5                     # halbe Mindestgeschwindigkeit
    trail, blade = fn.update_blade(trail, (100 + step, 100), DT)
    assert blade is None


def test_ruhender_stift_ist_kein_schnitt():
    trail, _ = fn.update_blade([], (100, 100), 0.0)
    for i in range(1, 10):
        trail, blade = fn.update_blade(trail, (100, 100), i * DT)
        assert blade is None


def test_sprung_gilt_als_fehlerkennung():
    trail, _ = fn.update_blade([], (100, 100), 0.0)
    trail, blade = fn.update_blade(trail, (100 + fn.MAX_JUMP + 50, 100), DT)
    assert blade is None
    assert len(trail) == 1


def test_alte_punkte_fallen_aus_der_spur():
    trail = []
    for i in range(30):
        trail, _ = fn.update_blade(trail, (100 + i * 30, 100), i * DT)
    now = 29 * DT
    assert all(now - t <= fn.TRAIL_TIME for _, t in trail)
    assert len(trail) < 30


def test_segment_trifft_frucht():
    assert fn.segment_hits((300, 300), (500, 300), 400, 320, 40)


def test_segment_verfehlt_frucht():
    assert not fn.segment_hits((300, 300), (500, 300), 400, 400, 40)


def test_segment_endet_vor_der_frucht():
    assert not fn.segment_hits((0, 0), (100, 0), 200, 0, 50)


def test_schnelles_durchziehen_trifft_obwohl_kein_frame_in_der_frucht_liegt():
    """Zwischen zwei Frames springt der Stift ueber die Frucht hinweg - trotzdem ein Treffer."""
    assert fn.segment_hits((200, 300), (600, 300), 400, 300, 30)


def test_punkt_segment_innerhalb_der_frucht():
    assert fn.segment_hits((400, 300), (400, 300), 400, 300, 30)
