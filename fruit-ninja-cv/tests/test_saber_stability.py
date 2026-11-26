"""Time-based visual stabilization, source fencing and hand compositing."""
import math
import cv2
import numpy as np
import pytest
import lightsaber as l
import config
from blade import Blade


def centre(s):return (np.array(s.hilt_top)+s.hilt_bottom)/2


def test_stationary_tip_and_grip_noise_is_reduced_without_stretching_handle():
    t=l.SaberTracker();raw=[];shown=[]
    for i in range(100):
        noise=np.array((3 if i%2 else -3,2 if i%3 else -2))
        tip=np.array((200,90))+noise
        grip=np.array((210,220))-noise
        s=t.update(tuple(tip),None,hands={'r':tuple(grip)},now=i/30,source=0)
        if i>20:raw.append(tip);shown.append(s.tip)
        assert abs(math.dist(s.hilt_top,s.hilt_bottom)-config.SABER_HILT_LENGTH)<1.5
        assert np.linalg.norm(centre(s)-grip)<9
    assert np.std(shown,axis=0).sum()<np.std(raw,axis=0).sum()*.7


def test_fast_coherent_movement_is_not_dragged_behind_the_hand():
    t=l.SaberTracker()
    t.update((200,90),None,hands={'r':(210,220)},now=0.,source=0)
    s=t.update((280,130),None,hands={'r':(290,260)},now=.033,source=0)
    assert math.dist(s.tip,(280,130))<=4
    assert np.linalg.norm(centre(s)-(290,260))<=5


def test_single_bad_glove_point_does_not_swing_weapon_across_the_screen():
    t=l.SaberTracker()
    first=t.update((200,90),None,hands={'r':(210,220)},now=0.,source=0)
    bad=t.update((200,90),None,hands={'r':(310,140)},now=.033,source=0)
    assert np.linalg.norm(centre(bad)-centre(first))<8
    back=t.update((200,90),None,hands={'r':(210,220)},now=.066,source=0)
    assert np.linalg.norm(centre(back)-centre(first))<5


def test_short_pose_dropout_holds_relative_grip_and_does_not_search_skin():
    t=l.SaberTracker();calls=[]
    a=t.update((200,90),None,hands={'r':(210,220)},now=0.,source=0)
    b=t.update((220,100),lambda:calls.append(1),hands={},now=.04,source=0)
    assert np.linalg.norm((centre(b)-centre(a))-(20,10))<5
    assert calls==[]


def test_marker_dropout_is_visual_only_and_expires_then_restarts_without_a_cut():
    t=l.SaberTracker();blade=Blade()
    for now,point in [(0.,(200,90)),(.033,(201,90)),(.066,None)]:
        blade.add(point);s=t.update(point,None,hands={'r':(210,220)},now=now,source=0)
    assert s is not None and 0<t.visibility<1
    assert blade.segment() is None and blade.current_point() is None
    assert t.update(None,None,now=.3,source=0) is None
    blade.add((300,90));t.update((300,90),None,hands={'r':(310,220)},now=.33,source=0)
    assert blade.segment() is None


def test_camera_change_discards_a_held_weapon_and_does_not_reuse_hand():
    t=l.SaberTracker()
    t.update((200,90),None,hands={'r':(210,220)},now=0.,source='a')
    assert t.update(None,None,now=.02,source='b') is None


def test_invalid_timestamp_clears_visual_history():
    t=l.SaberTracker();t.update((200,90),None,hands={'r':(210,220)},now=0.)
    assert t.update((200,90),None,now=float('nan')) is None
    assert t.update(None,None,now=.04) is None


def test_near_end_on_pen_does_not_flip_direction_when_points_cross():
    t=l.SaberTracker();directions=[]
    for i in range(30):
        s=t.update((200+i%2,200),None,hands={'r':(200-i%2,201)},now=i/30,source=0)
        d=np.array(s.hilt_bottom)-s.hilt_top;directions.append(d/np.linalg.norm(d))
    assert min(np.dot(a,b) for a,b in zip(directions,directions[1:]))>.95


@pytest.mark.parametrize('fps',[15,30,60])
def test_held_marker_visibility_expires_by_time_not_frame_count(fps):
    t=l.SaberTracker();t.update((200,90),None,hands={'r':(210,220)},now=0.,source=0)
    states=[(i/fps,t.update(None,None,now=i/fps,source=0)) for i in range(1,int(.3*fps)+1)]
    assert all(s is None for now,s in states if now>.15)


def test_current_hand_pixels_cover_the_middle_of_the_grip():
    frame=np.full((300,320,3),(35,55,75),np.uint8)
    cv2.rectangle(frame,(145,185),(175,225),(110,155,190),-1)
    hand=np.zeros(frame.shape[:2],np.uint8);hand[185:226,145:176]=255
    original=frame.copy();s=l.Saber((160,40),(160,175),(160,235))
    l.draw_saber(frame,s,config.SABER_COLOR,occlusion=(0,0,hand))
    assert np.max(np.abs(frame[205,160].astype(int)-original[205,160]))<5
    assert not np.array_equal(frame[181,160],original[181,160])
    assert np.array_equal(frame[:,0:50],original[:,0:50])


def test_beam_has_one_cyan_palette_without_pink_spill():
    frame=np.zeros((300,320,3),np.uint8)
    l.draw_saber(frame,l.Saber((160,40),(160,175),(160,235)),config.SABER_COLOR)
    b,g,r=map(int,frame[105,176])
    assert b>r+20 and g>r+20


def test_energy_shape_is_independent_of_baked_asset_halo(monkeypatch):
    import saber_art
    a=np.full((300,320,3),30,np.uint8);b=a.copy()
    saber=l.Saber((160,40),(160,175),(160,235))
    l.draw_saber(a,saber,config.SABER_COLOR)
    monkeypatch.setattr(saber_art,'load_parts',lambda:None)
    l.draw_saber(b,saber,config.SABER_COLOR)
    assert np.array_equal(a[:160],b[:160])


def test_alternating_skin_outliers_do_not_count_as_one_persistent_turn(monkeypatch):
    t=l.SaberTracker();t.update((200,90),None,now=0.)
    angle=[1.1]
    monkeypatch.setattr(t,'_target',lambda *args:(angle[0],140.))
    for i in range(20):
        angle[0]=1.1 if i%2 else -1.1
        s=t.update((200,90),None,now=(i+1)/30)
    assert abs(s.hilt_top[0]-200)<2


@pytest.mark.parametrize('fps',[15,30,60])
def test_tip_smoothing_follows_the_same_time_curve_at_different_frame_rates(fps):
    t=l.SaberTracker();t.update((200,90),None,hands={'r':(210,220)},now=0.)
    samples=[]
    for i in range(1,int(.2*fps)+1):
        s=t.update((208,90),None,hands={'r':(218,220)},now=i/fps)
        samples.append(s.tip)
    assert math.dist(samples[-1],(208,90))<2


def test_main_weapon_and_trail_are_cyan_even_when_tracking_pink(monkeypatch,tmp_path):
    from loop_harness import FakeCamera,green_frame,with_pen,run_loop
    calls=[]
    original=l.draw_saber
    def record(frame,saber,color,**kwargs):
        calls.append(color);return original(frame,saber,color,**kwargs)
    monkeypatch.setattr(l,'draw_saber',record)
    run_loop(monkeypatch,tmp_path,FakeCamera([with_pen(green_frame(),200)]),3,
             BG_MODE='off',SUIT_ENABLED=False)
    assert calls and all(color==config.SABER_COLOR for color in calls)


def test_marker_loss_in_the_real_loop_keeps_only_a_fading_visual(monkeypatch,tmp_path):
    from loop_harness import FakeCamera,green_frame,with_pen,run_loop
    import main as m
    draws=[];points=[]
    draw=l.draw_saber;track=m.MarkerTracker.track
    def recording_draw(frame,saber,color,**kwargs):
        draws.append((saber,kwargs.get('opacity')))
        return draw(frame,saber,color,**kwargs)
    def recording_track(tracker,frame):
        point=track(tracker,frame);points.append(point);return point
    monkeypatch.setattr(l,'draw_saber',recording_draw)
    monkeypatch.setattr(m.MarkerTracker,'track',recording_track)
    camera=FakeCamera([with_pen(green_frame(),200),green_frame()])
    now=[10.];read=camera.read
    def timed_read():now[0]+=.033;return read()
    monkeypatch.setattr(camera,'read',timed_read)
    monkeypatch.setattr(m.time,'monotonic',lambda:now[0])
    run_loop(monkeypatch,tmp_path,camera,2,
             BG_MODE='off',SUIT_ENABLED=False)
    assert points[0] is not None and points[1] is None
    assert draws[1][0] is not None and 0<draws[1][1]<1
