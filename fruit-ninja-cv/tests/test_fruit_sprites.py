"""Rendering contracts for the locally packaged fruit artwork."""
import math
import cv2
import numpy as np
import pytest

import config
import fruit_sprites as sprites
from fruit import Fruit


@pytest.mark.parametrize('name',[n for n,_ in config.FRUIT_TYPES]+['bomb'])
def test_assets_have_real_alpha_and_cached_size(name):
    image=sprites.get_sprite(name,30)
    assert image.shape[2]==4
    assert image[:,:,3].max()>=250 and image[0,0,3]==0
    assert np.count_nonzero(image[:,:,3])>400
    assert sprites.get_sprite(name,30) is image


def test_alpha_compositing_preserves_transparent_pixels_and_image_edges():
    raw=np.full((140,160,3),(21,38,59),np.uint8)
    out=raw.copy()
    assert sprites.draw(out,'apple',(80,70),30)
    assert np.array_equal(out[0],raw[0])
    assert np.linalg.norm(out[70,80].astype(float)-raw[70,80])>50
    sprites.draw(out,'bomb',(-10,5),30)
    assert np.array_equal(out[:,140:],raw[:,140:])
    before=out.copy();sprites.draw(out,'lime',(-300,5),30)
    assert np.array_equal(out,before)


@pytest.mark.parametrize('angle',[0,.7,math.pi/2,2.9])
def test_cut_at_zero_time_reconstructs_the_whole_sprite(angle):
    whole=np.full((180,180,3),27,np.uint8);cut=whole.copy()
    sprites.draw(whole,'orange',(90,90),32)
    sprites.draw(cut,'orange',(90,90),32,cut_angle=angle,spread=0)
    assert np.max(np.abs(whole.astype(int)-cut.astype(int)))<=1


def test_halves_separate_perpendicular_to_cut_and_reveal_background():
    out=np.full((180,180,3),27,np.uint8)
    sprites.draw(out,'apple',(90,90),30,cut_angle=0,spread=23)
    assert np.array_equal(out[90,90],(27,27,27))
    assert np.linalg.norm(out[65,90].astype(float)-27)>40
    assert np.linalg.norm(out[115,90].astype(float)-27)>40


def test_missing_asset_falls_back_to_procedural_fruit(monkeypatch):
    monkeypatch.setattr(sprites,'get_sprite',lambda *args:None)
    fruit=Fruit(200,160);fruit.name='apple';fruit.is_bomb=False
    fruit.x=100;fruit.y=80
    out=np.zeros((160,200,3),np.uint8)
    fruit.draw(out)
    assert out[80,100].any()


def test_game_entity_uses_artwork_for_whole_fruit_slices_and_bombs(monkeypatch):
    calls=[]
    def record(frame,name,center,radius,**kwargs):
        calls.append((name,center,radius,kwargs));return True
    monkeypatch.setattr(sprites,'draw',record)
    fruit=Fruit(200,160);fruit.name='apple';fruit.is_bomb=False
    frame=np.zeros((160,200,3),np.uint8)
    fruit.draw(frame)
    fruit.slice(.7);fruit.slice_time=.2;fruit.draw(frame)
    fruit.is_bomb=True;fruit.draw(frame)
    assert [c[0] for c in calls]==['apple','apple','bomb']
    assert calls[0][3]['cut_angle'] is None
    assert calls[1][3]['cut_angle']==.7
    assert calls[1][3]['spread']>0
