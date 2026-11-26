"""Concurrency contracts use events, not timing-sensitive sleeps."""
import threading
import numpy as np
import pytest
from inference_worker import LatestWorker


def test_busy_worker_keeps_latest_frame_and_discards_old_source():
    entered=threading.Event();release=threading.Event();finished=threading.Event()
    seen=[]
    def infer(frame):
        seen.append(int(frame[0,0]));entered.set()
        if len(seen)==1:assert release.wait(3)
        if len(seen)==2:finished.set()
        return frame.copy()
    w=LatestWorker()
    try:
        w.submit('yolo','a',np.full((2,2),1),infer,1.)
        assert entered.wait(3)
        for i in range(2,10):w.submit('yolo','a',np.full((2,2),i),infer,float(i))
        w.activate('yolo','b')
        assert w.poll('yolo','b') is None
        frame=np.full((2,2),42)
        w.submit('yolo','b',frame,infer,10.);frame[:]=99
        release.set();assert finished.wait(3)
        assert w.wait_idle(3)
        result=w.poll('yolo','b')
        assert seen==[1,42] and result.value[0,0]==42
        assert result.captured_at==10.
    finally:release.set();w.close()


def test_failure_is_once_and_close_never_waits_for_inference(caplog):
    count=[]
    def broken(frame):count.append(1);raise ValueError('fake net')
    w=LatestWorker()
    try:
        w.submit('pose','a',np.zeros((2,2)),broken,0.)
        assert w.wait_idle(3)
        for _ in range(10):w.submit('pose','a',np.zeros((2,2)),broken,1.)
        assert w.failed('pose') and len(count)==1
        assert len([r for r in caplog.records if 'Inference disabled' in r.message])==1
    finally:w.close()


def test_yolo_and_pose_never_run_concurrently_and_both_progress():
    w=LatestWorker();seen=[]
    def infer(frame):seen.append(threading.get_ident());return None
    try:
        w.submit('yolo','a',np.zeros((2,2)),infer,0.)
        w.submit('pose','a',np.zeros((2,2)),infer,0.)
        assert w.wait_idle(3)
        assert len(seen)==2 and len(set(seen))==1
        assert seen[0]!=threading.get_ident()
    finally:w.close()


def test_switch_away_and_back_cannot_resurrect_an_inflight_result():
    entered=threading.Event();release=threading.Event();w=LatestWorker()
    def infer(frame):entered.set();assert release.wait(3);return frame
    try:
        w.submit('yolo','camera-a',np.ones((2,2)),infer,1.)
        assert entered.wait(3)
        w.activate('yolo','camera-b');w.activate('yolo','camera-a')
        release.set();assert w.wait_idle(3)
        assert w.poll('yolo','camera-a') is None
    finally:release.set();w.close()


def test_error_from_a_cancelled_source_does_not_drop_the_new_sources_job():
    entered=threading.Event();release=threading.Event();w=LatestWorker()
    def bad(frame):entered.set();assert release.wait(3);raise ValueError('old camera')
    try:
        w.submit('pose','old',np.ones((2,2)),bad,0.)
        assert entered.wait(3)
        w.submit('pose','new',np.ones((2,2))*2,lambda frame:frame,1.)
        release.set();assert w.wait_idle(3)
        assert not w.failed('pose')
        assert w.poll('pose','new').value[0,0]==2
    finally:release.set();w.close()
