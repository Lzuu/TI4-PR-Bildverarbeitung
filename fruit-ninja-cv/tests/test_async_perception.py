"""Production async path, source identity and capture-time age contracts."""
import threading
import types
import cv2
import numpy as np
import pytest
import async_perception as a
import config
import main as m
import pose_tracking as p
import yolo_seg
from inference_worker import LatestWorker, Result
from loop_harness import FakeCamera, green_frame, run_loop
from test_pose_tracking import sample_pose
from test_yolo_seg import FakeNet, _blank_outputs


def test_real_game_loop_keeps_rendering_and_quits_while_yolo_forward_is_blocked(monkeypatch,tmp_path):
    entered=threading.Event();release=threading.Event()
    class SlowNet(FakeNet):
        def forward(self,names):
            entered.set();assert release.wait(5)
            return super().forward(names)
    net=SlowNet(*_blank_outputs())
    monkeypatch.setattr(yolo_seg,'create_segmenter',lambda:yolo_seg.BodySegmenter(net,input_size=64))
    workers=[]
    def worker():
        w=LatestWorker();workers.append(w);return w
    monkeypatch.setattr(m,'LatestWorker',worker)
    def on_frame(n,mouse):
        if n==1:assert entered.wait(3)
        assert not release.is_set()
    try:
        assert run_loop(monkeypatch,tmp_path,FakeCamera([green_frame()]),6,
                        ASYNC_INFERENCE=True,YOLO_ENABLED=True,BG_MODE='yolo',
                        on_frame=on_frame)==6
        assert workers[0]._closed
    finally:
        release.set()
        for w in workers:w._thread.join(3)


def test_async_mask_expires_from_capture_and_never_crosses_camera(monkeypatch):
    clock=[10.]
    monkeypatch.setattr(a.time,'monotonic',lambda:clock[0])
    w=LatestWorker();seg=a.AsyncSegmenter(w)
    monkeypatch.setattr(seg,'_infer',lambda frame,generation:np.full(frame.shape[:2],255,np.uint8))
    frame=np.zeros((20,20,3),np.uint8)
    try:
        assert seg.segment(frame,source=0) is None
        assert w.wait_idle(3)
        clock[0]=10.02
        assert seg.segment(frame,source=0).all()
        clock[0]=11.
        assert seg.segment(frame,source=0) is None
        assert seg.mask_age>=1.
        assert seg.segment(frame,source=1) is None
    finally:w.close()


def test_pose_uses_capture_age_and_aligns_the_inference_frame_before_flow(monkeypatch):
    class Infer:
        result=None
        def reset(self):pass
        def poll(self,*args):return self.result
    infer=Infer();tracker=p.PoseTracker(infer,max_age=.2)
    frame=np.zeros((200,160,3),np.uint8);mask=np.full((200,160),255,np.uint8)
    captured=np.full_like(frame,70)
    infer.result=Result(sample_pose(),captured,10.,.04,1)
    seen=[]
    monkeypatch.setattr(tracker,'_flow',lambda gray:seen.append(tracker._gray.copy()))
    assert tracker.update(frame,mask,now=10.1,source=0) is not None
    assert (seen[0]==70).all()
    infer.result=None
    assert tracker.update(frame,mask,now=10.21,source=0) is None


def test_pose_model_loading_is_off_the_game_thread_and_failure_is_latched(monkeypatch):
    called=[]
    def factory():
        called.append(threading.get_ident());return types.SimpleNamespace(infer=None)
    monkeypatch.setattr(p,'create_tracker',factory)
    monkeypatch.setattr(config,'POSE_ENABLED',True)
    w=LatestWorker();infer=a.AsyncPoseInfer(w)
    frame=np.zeros((200,160,3),np.uint8);mask=np.full((200,160),255,np.uint8)
    try:
        infer.poll(frame,mask,0.,0)
        assert w.wait_idle(3)
        infer.poll(frame,mask,1.,0)
        assert w.failed('pose') and len(called)==1
        assert called[0]!=threading.get_ident()
    finally:w.close()


def test_detector_only_runs_on_acquisition_and_after_pose_loss():
    class Net:
        confidence=1.
        def setInput(self,blob):pass
        def getUnconnectedOutLayersNames(self):return ['landmarks','score']
        def forward(self,names):
            raw=np.zeros((39,5),np.float32);raw[:,:2]=128;raw[:,3:]=10
            raw[34,:2]=(128,30)
            return [raw.reshape(1,-1),np.array([self.confidence])]
    calls=[]
    def detector(frame,mask,player):calls.append(1);return np.float32([40,40]),np.float32([40,5])
    net=Net();estimator=p.PoseEstimator(net,detector)
    frame=np.zeros((80,80,3),np.uint8);mask=np.full((80,80),255,np.uint8)
    for _ in range(8):assert estimator(frame,mask) is not None
    assert len(calls)==1
    net.confidence=0.;assert estimator(frame,mask) is None
    net.confidence=1.;assert estimator(frame,mask) is not None
    assert len(calls)==2
    assert estimator(frame,np.zeros_like(mask)) is None
    estimator(frame,mask)
    assert len(calls)==3


def test_async_real_loop_reports_mask_and_pose_age_and_inference_cost(monkeypatch,tmp_path,caplog):
    import logging
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(yolo_seg,'create_segmenter',lambda:types.SimpleNamespace(
        _infer=lambda frame:np.full(frame.shape[:2],255,np.uint8)))
    class Estimator:
        failed=False
        def reset(self):pass
        def __call__(self,frame,mask):return sample_pose()
    monkeypatch.setattr(p,'create_tracker',lambda:p.PoseTracker(Estimator()))
    monkeypatch.setattr(m.suit,'dress',lambda frame,*args,**kwargs:frame)
    workers=[]
    def worker():
        w=LatestWorker();workers.append(w);return w
    monkeypatch.setattr(m,'LatestWorker',worker)
    def on_frame(n,mouse):assert workers[0].wait_idle(3)
    try:
        run_loop(monkeypatch,tmp_path,FakeCamera([green_frame()]),5,
                 ASYNC_INFERENCE=True,YOLO_ENABLED=True,POSE_ENABLED=True,
                 SUIT_ENABLED=True,BG_MODE='yolo',PERF_LOG_INTERVAL=0.,on_frame=on_frame)
        lines=[r.getMessage() for r in caplog.records if r.getMessage().startswith('Perf ')]
        for name in ('mask_age_ms','pose_age_ms','yolo_infer','pose_infer'):
            assert any(name in line for line in lines),name
    finally:
        for w in workers:w.close();w._thread.join(3)


# -- #88: the YOLO mask must not disappear every other frame at a low frame rate -----------
class _FakeWorker:
    """What the segmenter needs from the worker; a submit is answered one frame later."""

    def __init__(self, mask_shape):
        self.mask = np.full(mask_shape, 255, np.uint8)
        self.captured = None                   # capture time of the newest finished inference
        self.pending = None
        self.dues = []
        self._sequence = 0

    def activate(self, kind, key):
        pass

    def failed(self, kind):
        return False

    def poll(self, kind, key):
        if self.pending is not None:           # the inference finished before this frame
            self.captured, self.pending = self.pending, None
            self._sequence += 1
        if self.captured is None:
            return None
        return Result(self.mask, None, self.captured, .05, self._sequence)

    def submit(self, kind, key, frame, infer, captured_at, due=True):
        self.dues.append(due)
        if due:
            self.pending = captured_at


def _mask_availability(monkeypatch, interval, frames):
    clock = [10.]
    monkeypatch.setattr(a.time, 'monotonic', lambda: clock[0])
    worker = _FakeWorker((20, 20))
    seg = a.AsyncSegmenter(worker)
    seg.state = seg.READY
    frame = np.zeros((20, 20, 3), np.uint8)
    got = []
    for _ in range(frames):
        got.append(seg.segment(frame, source=0) is not None)
        clock[0] += interval
    return got, worker.dues


def test_mask_is_there_in_every_frame_at_three_frames_per_second(monkeypatch):
    # inference every third frame (0.9 s) with a 0.3 s age limit left two of three frames
    # without any mask: no space background, no suit -- the "flicker"
    got, dues = _mask_availability(monkeypatch, .35, 20)
    assert all(got[6:]), got
    assert all(dues[6:])                         # every frame is a new inference now


def test_cadence_is_unchanged_at_a_normal_frame_rate(monkeypatch):
    got, dues = _mask_availability(monkeypatch, 1 / 30, 36)
    assert all(got[6:])
    assert dues[12:] == [i % 3 == 0 for i in range(12, 36)]  # every third frame, as before


def test_cadence_speeds_up_gradually_when_the_game_slows_down(monkeypatch):
    _, fast = _mask_availability(monkeypatch, 1 / 30, 30)
    _, medium = _mask_availability(monkeypatch, .1, 30)
    _, slow = _mask_availability(monkeypatch, .35, 30)
    assert sum(fast) < sum(medium) <= sum(slow)
