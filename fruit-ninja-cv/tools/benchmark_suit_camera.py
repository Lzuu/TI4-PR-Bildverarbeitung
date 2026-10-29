"""Local USB-camera throughput check. No frames/faces are saved or uploaded."""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import cameras,face_id,yolo_seg,background
from main import crop_to_aspect
from marker_tracker import MarkerTracker
from pose_tracking import create_tracker
from suit_rig import SuitRenderer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera',type=int,default=2)
    parser.add_argument('--seconds',type=float,default=8)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    camera=cameras.Camera()
    result=dict(camera=args.camera,camera_opened=False,frames_saved=False,display=False)
    try:
        if not camera.open_first([args.camera]):
            result['reason']='No readable device at requested camera index'
            print(json.dumps(result));return 2
        result['camera_opened']=True
        paths=face_id.ensure_models(download=False)
        identifier=face_id.FaceIdentifier(paths['detector'],paths['recognizer'])
        segmenter=yolo_seg.BodySegmenter(cv2.dnn.readNetFromONNX(yolo_seg.ensure_model(download=False)))
        pose=create_tracker(download=False)
        renderer=SuitRenderer();marker=MarkerTracker()
        backdrop=background._generate_space(960,540)
        elapsed=[];poses=faces=markers=failures=0
        start=time.perf_counter()
        while time.perf_counter()-start<args.seconds:
            tick=time.perf_counter()
            ok,frame=camera.read()
            if not ok:
                failures+=1
                if failures>=3:break
                continue
            clean=cv2.resize(crop_to_aspect(frame,16/9),(960,540))
            markers+=int(marker.track(clean) is not None)
            mask=segmenter.segment(clean,source=args.camera)
            box=identifier.largest_face_box(clean);faces+=int(box is not None)
            body=pose.update(clean,mask,source=args.camera);poses+=int(body is not None)
            canvas=background.composite_space(clean.copy(),mask,backdrop,feather=1.5)
            renderer.render(canvas,clean,mask,body,box,marker.last_mask,source=args.camera)
            elapsed.append((time.perf_counter()-tick)*1000)
        duration=time.perf_counter()-start
        result.update(duration_s=round(duration,3),frames=len(elapsed),read_failures=failures,
                      live_processing_fps=round(len(elapsed)/duration,2),
                      frames_with_pose=poses,frames_with_face=faces,frames_with_marker=markers)
        if elapsed:
            result.update(median_frame_ms=round(float(np.median(elapsed)),3),
                          p95_frame_ms=round(float(np.percentile(elapsed,95)),3))
        result['limitations']='960x540 local processing including capture, without display. Counts show whether a player/pen was actually present; no motion scenario is automatically certified.'
        print(json.dumps(result,indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        return 0
    finally:
        camera.release()


if __name__=='__main__':raise SystemExit(main())
