"""Generated laser-sword rendering keeps the existing tip/emitter/grip anchors."""
import cv2
import numpy as np
import pytest
import lightsaber
import saber_art


@pytest.mark.parametrize('points',[
    ((120,40),(120,190),(120,245)),
    ((40,100),(190,100),(245,100)),
    ((230,40),(130,170),(95,215)),
])
def test_artwork_parts_map_to_the_actual_weapon_pose(points):
    parts=saber_art.load_parts()
    assert parts is saber_art.load_parts()
    saber=lightsaber.Saber(*points)
    for name,start,end in [('hilt',saber.hilt_top,saber.hilt_bottom)]:
        sprite,anchors,width=parts[name]
        matrix=saber_art.part_matrix(anchors,start,end,width,24)
        assert np.allclose(cv2.transform(anchors[None],matrix)[0],(start,end),atol=1e-4)
        assert sprite.image.shape[2]==4 and sprite.image[0,0,3]==0


def test_generated_saber_uses_alpha_and_clips_offscreen():
    frame=np.full((300,320,3),28,np.uint8);before=frame.copy()
    saber=lightsaber.Saber((160,30),(160,180),(160,245))
    assert saber_art.draw(frame,saber,1.)
    assert np.array_equal(frame[:,0:30],before[:,0:30])
    assert np.any(frame[190:235,150:170]!=28)
    assert saber_art.draw(frame,lightsaber.Saber((-5,-80),(-5,70),(-5,130)),1.)


def test_missing_artwork_keeps_the_existing_weapon_visible(monkeypatch):
    monkeypatch.setattr(saber_art,'load_parts',lambda:None)
    frame=np.zeros((300,320,3),np.uint8)
    saber=lightsaber.Saber((160,30),(160,180),(160,245))
    lightsaber.draw_saber(frame,saber,(255,0,0))
    assert frame[90,160].min()>200
    assert frame[210,160].any()
