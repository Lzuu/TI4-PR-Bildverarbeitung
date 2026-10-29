"""Installer and runtime-failure contracts, without network or model dependencies."""
from io import BytesIO
import cv2
import numpy as np
import pytest
import pose_tracking as p


def test_corrupt_download_is_not_installed_or_left_as_an_active_cache(monkeypatch,tmp_path):
    monkeypatch.setattr(p.urllib.request,'urlopen',lambda *a,**k:BytesIO(b'invalid onnx'))
    with pytest.raises(RuntimeError,match='hash mismatch'):
        p.ensure_model(tmp_path)
    assert not list(tmp_path.iterdir())


def test_existing_corrupt_model_is_reported_without_overwriting_it(tmp_path):
    path=tmp_path/p.MODEL_NAME;path.write_bytes(b'broken')
    with pytest.raises(RuntimeError,match='hash mismatch'):
        p.ensure_model(tmp_path,download=False)
    assert path.read_bytes()==b'broken'


def test_unavailable_pose_model_is_a_limited_fallback(monkeypatch):
    monkeypatch.setattr(p.config,'POSE_ENABLED',True)
    monkeypatch.setattr(p,'ensure_model',lambda **kwargs:(_ for _ in ()).throw(OSError('review fake')))
    tracker=p.create_tracker(download=False)
    assert tracker.infer is None


def test_inference_failure_disables_network_instead_of_retrying_every_frame():
    class BrokenNet:
        calls=0
        def setInput(self,blob):pass
        def getUnconnectedOutLayersNames(self):return ['out']
        def forward(self,names):
            self.calls+=1
            raise cv2.error('fake inference failure')
    net=BrokenNet();estimator=p.PoseEstimator(net)
    frame=np.zeros((80,80,3),np.uint8);mask=np.full((80,80),255,np.uint8)
    assert estimator(frame,mask) is None
    assert estimator(frame,mask) is None
    assert net.calls==1 and estimator.failed
