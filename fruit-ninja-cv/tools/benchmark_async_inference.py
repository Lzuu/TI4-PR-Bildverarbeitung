"""Offline comparison of inference-facing UI work; not camera FPS.

python tools/benchmark_async_inference.py --frames 100
Uses the public NASA test portrait and already installed models only.
"""
from pathlib import Path
import argparse
import json
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import config
import pose_tracking as p
import yolo_seg as y
from async_perception import AsyncSegmenter, AsyncPoseInfer
from inference_worker import LatestWorker


def models():
    segmenter=y.BodySegmenter(cv2.dnn.readNetFromONNX(y.ensure_model(download=False)))
    detector=p.PersonDetector(cv2.dnn.readNetFromONNX(p.ensure_model(download=False,kind='detector')))
    calls=[]
    def detect(frame,mask):calls.append(1);return detector(frame,mask)
    estimator=p.PoseEstimator(cv2.dnn.readNetFromONNX(p.ensure_model(download=False)),detect)
    return segmenter,p.PoseTracker(estimator),calls


def run(frame,count,asynchronous):
    detector_calls=[]
    def load():
        seg,tracker,calls=models();detector_calls.append(calls);return seg,tracker
    worker=None
    if asynchronous:
        # Factories themselves run inside the worker, never in the timed UI path.
        loaded={}
        def resources():
            if not loaded:loaded['models']=load()
            return loaded['models']
        old_seg,old_pose=y.create_segmenter,p.create_tracker
        y.create_segmenter=lambda:resources()[0]
        p.create_tracker=lambda:resources()[1]
        worker=LatestWorker();segmenter=AsyncSegmenter(worker)
        tracker=p.PoseTracker(AsyncPoseInfer(worker))
    else:
        segmenter,tracker=load()
    values=[];masks=poses=0;ages=[]
    try:
        # Warm load/inference and optical flow, not included in timing.
        for i in range(15):
            mask=segmenter.segment(frame,source=0)
            if mask is not None:tracker.update(frame,mask,source=0)
            if worker:worker.wait_idle(10)
            time.sleep(.034)
        for i in range(count):
            start=time.perf_counter()
            mask=segmenter.segment(frame,source=0)
            if mask is not None:
                masks+=1
                pose=tracker.update(frame,mask,source=0)
                poses+=pose is not None
                ages.append(segmenter.mask_age*1000.)
            values.append((time.perf_counter()-start)*1000.)
            time.sleep(max(0.,1/30-(time.perf_counter()-start)))
        return dict(mode='async' if asynchronous else 'sync',frames=count,
                    median_ui_ms=round(statistics.median(values),3),
                    p95_ui_ms=round(float(np.percentile(values,95)),3),
                    max_ui_ms=round(max(values),3),frames_with_mask=masks,
                    frames_with_pose=int(poses),max_mask_age_ms=round(max(ages,default=0),3),
                    detector_calls=sum(len(x) for x in detector_calls))
    finally:
        if worker:
            worker.close();worker._thread.join(10)
            y.create_segmenter,p.create_tracker=old_seg,old_pose


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=int,default=100)
    args=parser.parse_args()
    cv2.setNumThreads(config.INFERENCE_CV_THREADS)
    image=cv2.imread(str(ROOT/'tests/data/armstrong_portrait.jpg'))
    results=[run(image,args.frames,False),run(image,args.frames,True)]
    out=dict(opencv=cv2.__version__,opencv_threads=cv2.getNumThreads(),
             image_size=list(image.shape[:2]),camera=False,render=False,display=False,
             note='Warm static portrait; timed segment + pose/update on UI only. No live FPS claim.',
             results=results)
    destination=ROOT.parent/'docs/async-inference-56';destination.mkdir(parents=True,exist_ok=True)
    (destination/'measurements.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
