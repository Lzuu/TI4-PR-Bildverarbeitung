"""Teilaufgabe 5: Spiellogik (Fruechte, Bomben, Wuerfe, Highscore)."""
import random

import fruit_ninja as fn


def fly(fruit, dt=1 / 30, max_time=10):
    """Simuliert den Flug und liefert (Flugzeit, hoechster Punkt)."""
    t, top = 0.0, fruit.y
    while not fruit.gone() and t < max_time:
        fruit.update(dt)
        top = min(top, fruit.y)
        t += dt
    return t, top


def test_frucht_startet_unter_dem_bild_und_fliegt_hinein():
    random.seed(1)
    for _ in range(50):
        f = fn.spawn_fruit()
        assert f.y > fn.H
        assert fn.W * 0.15 <= f.x <= fn.W * 0.85
        t, top = fly(f)
        assert t < 10, "Frucht faellt nie wieder heraus"
        assert top < fn.H * 0.5, "Frucht kommt nicht hoch genug ins Bild"
        assert top > 0, "Frucht fliegt oben aus dem Bild"


def test_bombe_wird_als_bombe_erzeugt():
    assert isinstance(fn.spawn_fruit(bomb=True), fn.Bomb)
    assert not isinstance(fn.spawn_fruit(), fn.Bomb)


def test_zerschneiden_erzeugt_zwei_haelften_und_saft():
    effects = []
    fn.slice_fruit(fn.spawn_fruit(), effects)
    assert sum(isinstance(e, fn.Half) for e in effects) == 2
    assert sum(isinstance(e, fn.Particle) for e in effects) > 0


def test_bombe_explodiert_mit_hinweis():
    effects = []
    fn.explode_bomb(fn.spawn_fruit(bomb=True), effects)
    assert any(isinstance(e, fn.Popup) for e in effects)
    assert not any(isinstance(e, fn.Half) for e in effects)


def test_wurf_planung_liefert_gueltige_werte():
    rng = random.Random(0)
    for i in range(1000):
        count, bomb, pause = fn.plan_wave(i / 1000, rng.uniform(*fn.ROUND_INTENSITY), rng)
        assert 1 <= count <= 6
        assert isinstance(bomb, bool)
        assert pause > 0


def test_runden_haben_unterschiedlich_viele_fruechte():
    """Anforderung: nicht in jeder Runde gleich viele Fruechte."""
    rng = random.Random(0)
    totals = set()
    for _ in range(30):
        intensity = rng.uniform(*fn.ROUND_INTENSITY)
        t = total = 0
        while t < fn.ROUND_TIME:
            count, _, pause = fn.plan_wave(t / fn.ROUND_TIME, intensity, rng)
            total += count
            t += pause
        totals.add(total)
    assert len(totals) > 10


def test_bomben_werden_gegen_rundenende_haeufiger():
    rng = random.Random(0)
    start = sum(fn.plan_wave(0.0, 1.0, rng)[1] for _ in range(5000))
    end = sum(fn.plan_wave(1.0, 1.0, rng)[1] for _ in range(5000))
    assert end > start * 1.5


def test_highscore_fehlende_datei_ergibt_null(tmp_path, monkeypatch):
    monkeypatch.setattr(fn, "HIGHSCORE_FILE", str(tmp_path / "gibtsnicht.txt"))
    assert fn.load_highscore() == 0


def test_highscore_speichern_und_laden(tmp_path, monkeypatch):
    monkeypatch.setattr(fn, "HIGHSCORE_FILE", str(tmp_path / "highscore.txt"))
    fn.save_highscore(42)
    assert fn.load_highscore() == 42


def test_kaputte_highscore_datei_ergibt_null(tmp_path, monkeypatch):
    path = tmp_path / "highscore.txt"
    path.write_text("kein score")
    monkeypatch.setattr(fn, "HIGHSCORE_FILE", str(path))
    assert fn.load_highscore() == 0
