"""Reproducible headless stage breakdown for the articulated suit; no models."""
from pathlib import Path
import argparse,json,statistics,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import cv2
import numpy as np
import config
from perf import FrameStats
from suit_rig import SuitRenderer
from preview_suit import scene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=60)
    parser.add_argument('--threads',type=int,default=config.INFERENCE_CV_THREADS)
    args=parser.parse_args()
    cv2.setNumThreads(max(1,args.threads))
    raw,mask,pose,face,keep=scene('side');renderer=SuitRenderer()
    stats=FrameStats(interval=0,log=None)
    for _ in range(8):renderer.render(raw.copy(),raw,mask,pose,face,keep,now=.7)
    values={False:[],True:[]};details={}
    for i in range(args.iterations):
        # Alternate order to reduce warm-up / CPU-frequency bias.
        for enabled in ((False,True) if i%2==0 else (True,False)):
            frame=raw.copy();stats.start_frame()
            start=time.perf_counter()
            renderer.render(frame,raw,mask,pose,face,keep,now=.7,stats=stats if enabled else None)
            values[enabled].append((time.perf_counter()-start)*1000.)
            summary=stats.end_frame()
            if enabled:
                for name,data in summary['stages'].items():
                    details.setdefault(name,[]).append(data['median'])
    describe=lambda items:dict(median_ms=round(statistics.median(items),3),p95_ms=round(float(np.percentile(items,95)),3))
    result=dict(opencv=cv2.__version__,opencv_threads=cv2.getNumThreads(),
                frame=[960,720],pose='side',iterations=args.iterations,
                camera=False,display=False,models=False,
                render_without_sink=describe(values[False]),render_with_sink=describe(values[True]),
                stages={name:describe(items) for name,items in details.items()},
                note='Warm synthetic render only; parent suit time includes all substeps, do not add it twice.')
    dest=ROOT.parent/'docs/suit-timings-62';dest.mkdir(parents=True,exist_ok=True)
    (dest/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
