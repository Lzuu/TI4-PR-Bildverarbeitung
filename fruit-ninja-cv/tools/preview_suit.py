"""Reproducible public/synthetic previews and headless timings (no webcam).

python tools/preview_suit.py --models --benchmark
All faces come from the public-domain NASA portrait already in tests/data.
"""
from pathlib import Path
import argparse
import json
import platform
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np

from pose_tracking import Pose,PoseEstimator,PersonDetector,ensure_model
from suit_rig import SuitRenderer,load_assets
from marker_tracker import MarkerTracker
import background
import face_id
import yolo_seg


def scene(kind='down',scale=1.,shift=(0,0)):
    h,w=720,960
    yy,xx=np.mgrid[:h,:w]
    raw=np.empty((h,w,3),np.uint8)
    raw[:,:,0]=(42+18*yy/h).astype(np.uint8)
    raw[:,:,1]=(27+12*xx/w).astype(np.uint8)
    raw[:,:,2]=(18+8*yy/h).astype(np.uint8)
    mask=np.zeros((h,w),np.uint8)
    p=np.zeros((33,3),np.float32)
    confidence=np.zeros(33,np.float32)
    xy={11:(390,245),12:(570,245),13:(370,335),14:(590,335),15:(365,425),16:(595,425),
        19:(363,455),20:(597,455),23:(420,420),24:(540,420),25:(420,530),26:(540,530),
        27:(420,645),28:(540,645),31:(414,674),32:(546,674)}
    if kind=='side':xy.update({13:(300,245),14:(660,245),15:(210,245),16:(750,245),19:(180,245),20:(780,245)})
    if kind=='up':xy.update({13:(330,165),14:(630,165),15:(360,65),16:(600,65),19:(370,35),20:(590,35)})
    if kind=='cross':xy.update({13:(370,340),14:(590,340),15:(555,305),16:(405,295),19:(585,298),20:(375,288)})
    for index,point in xy.items():
        p[index,:2]=point;confidence[index]=1
    face=np.float32([447,125,66,90])
    for index,point in {0:(480,177),2:(492,159),5:(466,159),7:(510,165),8:(450,165),9:(490,194),10:(469,194)}.items():
        p[index,:2]=point;confidence[index]=1
    center=np.float32([480,125])
    p[:,:2]=(p[:,:2]-center)*scale+center+shift
    face[:2]=(face[:2]-center)*scale+center+shift
    face[2:]*=scale
    if kind=='cross':p[[13,15,19,14,16,20],2]=-45
    def pos(i):return tuple(np.rint(p[i,:2]).astype(int))
    def bone(a,b,width):
        cv2.line(mask,pos(a),pos(b),255,max(2,round(width*scale)),cv2.LINE_AA)
        cv2.line(raw,pos(a),pos(b),(65,72,86),max(2,round(width*scale)),cv2.LINE_AA)
    torso=np.rint(p[[11,12,24,23],:2]).astype(np.int32)
    cv2.fillConvexPoly(mask,torso,255)
    cv2.fillConvexPoly(raw,torso,(72,78,92))
    # A real pelvis joins the thighs below the measured hip-centre line.
    hip_left=np.rint(p[23,:2]).astype(int);hip_right=np.rint(p[24,:2]).astype(int)
    pelvis=np.array([hip_left,hip_right,hip_right+(0,round(28*scale)),
                     hip_left+(0,round(28*scale))],np.int32)
    cv2.fillConvexPoly(mask,pelvis,255);cv2.fillConvexPoly(raw,pelvis,(72,78,92))
    for a,b,width in [(23,25,55),(24,26,55),(25,27,45),(26,28,45),(27,31,48),(28,32,48),
                       (11,13,48),(12,14,48),(13,15,40),(14,16,40),(15,19,36),(16,20,36)]:bone(a,b,width)
    fx,fy,fw,fh=map(round,face)
    head_center=(round(fx+fw/2),round(fy+fh/2))
    cv2.ellipse(mask,head_center,(round(fw*.62),round(fh*.65)),0,0,360,255,-1)
    # Neck keeps head and torso in one connected component.
    neck_bottom=((pos(11)[0]+pos(12)[0])//2,(pos(11)[1]+pos(12)[1])//2)
    cv2.line(mask,head_center,neck_bottom,255,max(4,round(fw*.65)))
    cv2.line(raw,head_center,neck_bottom,(85,120,160),max(4,round(fw*.65)))
    portrait=cv2.imread(str(ROOT/'tests/data/armstrong_portrait.jpg'))
    face_pixels=cv2.resize(portrait[100:236,248:340],(max(fw,1),max(fh,1)))
    x0,y0,x1,y1=max(fx,0),max(fy,0),min(fx+fw,w),min(fy+fh,h)
    if x1>x0 and y1>y0:raw[y0:y1,x0:x1]=face_pixels[y0-fy:y1-fy,x0-fx:x1-fx]
    keep=np.zeros_like(mask)
    wrist=pos(16)
    cv2.line(keep,(wrist[0]+5,wrist[1]-38),(wrist[0]+5,wrist[1]+8),255,max(3,round(8*scale)))
    raw[keep>0]=(180,0,255)
    # The person network can exclude the separate controller.
    mask[keep>0]=0
    outside=(p[:,0]<-10)|(p[:,0]>w+10)|(p[:,1]<-10)|(p[:,1]>h+10)
    confidence[outside]=0
    return raw,mask,Pose(p,confidence),tuple(face),keep


def label(image,text):
    cv2.rectangle(image,(0,0),(image.shape[1],30),(28,25,23),-1)
    cv2.putText(image,text,(12,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,225,230),1,cv2.LINE_AA)
    return image


def measure(name,fn,iterations=40):
    for _ in range(3):fn()
    values=[]
    for _ in range(iterations):
        start=time.perf_counter_ns();fn();values.append((time.perf_counter_ns()-start)/1e6)
    return dict(stage=name,iterations=iterations,median_ms=round(statistics.median(values),3),
                p95_ms=round(float(np.percentile(values,95)),3),mean_ms=round(statistics.mean(values),3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT.parent/'docs/suit-previews')
    parser.add_argument('--models',action='store_true',help='Run the already downloaded real models offline')
    parser.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];metrics=[]
    cases=[('down','Arme unten',1.,(0,0)),('side','Arme seitlich',1.,(0,0)),
           ('up','Arme ueber Kopf',1.,(0,0)),('cross','Arme vor dem Koerper',1.,(0,0)),
           ('down','Nah / Beine abgeschnitten',1.4,(0,0)),('side','Weiter entfernt',.65,(0,0)),
           ('side','Am linken Bildrand',1.,(-420,0))]
    outputs={}
    for index,(kind,title,scale,shift) in enumerate(cases):
        raw,mask,pose,face,keep=scene(kind,scale,shift)
        renderer=SuitRenderer()
        canvas=raw.copy()
        out=renderer.render(canvas,raw,mask,pose,face,keep,now=0,source=index)
        outputs[kind if index<4 else str(index)]=out
        rows.append(label(cv2.resize(out,(384,288)),title))
    raw,mask,pose,face,keep=scene()
    r=SuitRenderer();r.render(raw.copy(),raw,mask,pose,face,keep,now=0)
    held=r.render(raw.copy(),raw,mask,pose=pose,face=None,keep=keep,now=.05)
    expired=r.render(raw.copy(),raw,mask,pose=None,face=None,keep=keep,now=.5)
    rows.extend([label(cv2.resize(held,(384,288)),'Kurzer Gesichtsausfall / Pose folgt'),
                 label(cv2.resize(expired,(384,288)),'Ausfall: keine schwebenden Teile')])
    contact=np.vstack([np.hstack(rows[i:i+3]) for i in range(0,9,3)])
    cv2.imwrite(str(args.output/'synthetic-poses.png'),contact)
    close=cv2.resize(outputs['down'][55:270,400:560],(480,645))
    cv2.imwrite(str(args.output/'clear-visor.png'),close)
    gallery=[]
    assets,_=load_assets()
    for name,sprite in assets.items():
        source=sprite.image
        tile=np.full((256,256,3),45,np.uint8)
        image=cv2.resize(source,(230,230),interpolation=cv2.INTER_AREA)
        alpha=image[:,:,3:4]/255.
        tile[26:256,13:243]=(image[:,:,:3]*alpha+tile[26:256,13:243]*(1-alpha)).astype(np.uint8)
        gallery.append(label(tile,name))
    cv2.imwrite(str(args.output/'asset-parts.png'),np.vstack((np.hstack(gallery[:4]),np.hstack(gallery[4:]))))
    if args.benchmark:
        renderer=SuitRenderer()
        metrics.append(measure('articulated_render_960x720',lambda:renderer.render(raw.copy(),raw,mask,pose,face,keep,now=0)))
        tracker=MarkerTracker()
        metrics.append(measure('marker_960x720',lambda:tracker.track(raw)))
    real={}
    if args.models:
        paths=face_id.ensure_models(download=False)
        identifier=face_id.FaceIdentifier(paths['detector'],paths['recognizer'])
        net=cv2.dnn.readNetFromONNX(yolo_seg.ensure_model(download=False))
        segmenter=yolo_seg.BodySegmenter(net,every_n=1)
        detector=PersonDetector(cv2.dnn.readNetFromONNX(ensure_model(download=False,kind='detector')))
        estimator=PoseEstimator(cv2.dnn.readNetFromONNX(ensure_model(download=False)),detector)
        portrait=cv2.imread(str(ROOT/'tests/data/armstrong_portrait.jpg'))
        real_mask=segmenter.segment(portrait)
        real_pose=estimator(portrait,real_mask)
        real_face=identifier.largest_face_box(portrait)
        renderer=SuitRenderer()
        real_out=renderer.render(portrait.copy(),portrait,real_mask,real_pose,real_face,now=0)
        before=label(portrait.copy(),'Original NASA-Testbild / public domain')
        after=label(real_out.copy(),'Echter ONNX-Pose-/YOLO-/YuNet-Lauf')
        cv2.imwrite(str(args.output/'public-portrait.png'),np.hstack((before,after)))
        space=background._generate_space(portrait.shape[1],portrait.shape[0])
        composed=background.composite_space(portrait.copy(),real_mask,space,feather=1.5)
        space_out=SuitRenderer().render(composed,portrait,real_mask,real_pose,real_face,now=0)
        cv2.imwrite(str(args.output/'public-space.png'),space_out)
        real=dict(pose_detected=real_pose is not None,
                  confident_joints=int(np.sum(real_pose.confidence>=.5)) if real_pose else 0,
                  face_box=real_face,pose_failed=estimator.failed)
        if args.benchmark:
            metrics.extend([measure('blazepose_256_on_public_portrait',lambda:estimator(portrait,real_mask)),
                            measure('yolo_320_on_public_portrait',lambda:segmenter.segment(portrait)),
                            measure('yunet_320_on_public_portrait',lambda:identifier.largest_face_box(portrait))])
            def pipeline():
                m=segmenter.segment(portrait)
                p=estimator(portrait,m)
                f=identifier.largest_face_box(portrait)
                return renderer.render(portrait.copy(),portrait,m,p,f,now=0)
            metrics.append(measure('all_models_and_render_public_portrait_each_frame',pipeline))
    metadata=dict(environment=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__,
                                    opencv_threads=cv2.getNumThreads()),
                  camera=False,display=False,real_model_smoke=real,timings=metrics,
                  limitations='Headless warm measurements; synthetic poses validate rendering, not pose-model accuracy. No live FPS or input-to-display latency measured.')
    (args.output/'measurements.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
