"""Compare OpenCV CPU thread budgets with the same public portrait and models."""
from pathlib import Path
import json
import argparse
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import background,face_id,yolo_seg
from pose_tracking import PoseEstimator,PersonDetector,PoseTracker,ensure_model
from suit_rig import SuitRenderer
from marker_tracker import MarkerTracker
from preview_suit import measure


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--threads',type=int,nargs='+',default=[16,4,2,1])
arguments=parser.parse_args()
frame=cv2.imread(str(ROOT/'tests/data/armstrong_portrait.jpg'))
paths=face_id.ensure_models(download=False)
face=face_id.FaceIdentifier(paths['detector'],paths['recognizer'])
pose=PoseEstimator(cv2.dnn.readNetFromONNX(ensure_model(download=False)),
                   PersonDetector(cv2.dnn.readNetFromONNX(ensure_model(download=False,kind='detector'))))
seg=yolo_seg.BodySegmenter(cv2.dnn.readNetFromONNX(yolo_seg.ensure_model(download=False)),every_n=3)
tracker=PoseTracker(pose)
renderer=SuitRenderer()
marker=MarkerTracker()
results=[]
for threads in arguments.threads:
    cv2.setNumThreads(threads)
    def pipeline():
        mask=seg.segment(frame)
        point=marker.track(frame)
        now=time.monotonic()
        body=tracker.update(frame,mask,now=now)
        box=face.largest_face_box(frame)
        return renderer.render(frame.copy(),frame,mask,body,box,marker.last_mask,now)
    row=measure('cadenced_pipeline_public_portrait',pipeline,iterations=24)
    row['opencv_threads']=threads
    results.append(row)
    print(json.dumps(row),flush=True)
out=ROOT.parent/'docs/suit-previews'/('cadenced-pipeline.json' if len(arguments.threads)==1 else 'thread-comparison.json')
out.write_text(json.dumps(dict(opencv=cv2.__version__,numpy=__import__('numpy').__version__,
                              image_size=[480,600],camera=False,display=False,
                              limitations='Static public portrait; short headless series, no live FPS or thermal endurance test.',
                              measurements=results),indent=2),encoding='utf-8')
