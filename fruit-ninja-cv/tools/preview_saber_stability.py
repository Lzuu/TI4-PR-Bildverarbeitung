"""Synthetic hand/contact and motion replay; no webcam, GUI or model inference.

Optional before-image uses the repository's own historical renderer via git show.
"""
from pathlib import Path
import argparse,json,math,statistics,subprocess,sys,time,types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import cv2
import numpy as np
import config
import lightsaber
from suit_rig import SuitRenderer
from preview_suit import scene


def historical(ref):
    modules=[]
    try:
        for name in ('saber_art','lightsaber'):
            source=subprocess.check_output(['git','show',ref+':fruit-ninja-cv/'+name+'.py'],cwd=ROOT.parent,text=True,stderr=subprocess.DEVNULL)
            module=types.ModuleType('before_'+name);module.__file__=str(ROOT/(name+'.py'))
            exec(compile(source,module.__file__,'exec'),module.__dict__);modules.append(module)
        modules[1].saber_art=modules[0]
        return modules[1]
    except (OSError,subprocess.CalledProcessError):return None


def label(image,title):
    cv2.rectangle(image,(0,0),(image.shape[1],30),(25,24,22),-1)
    cv2.putText(image,title,(12,21),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1,cv2.LINE_AA)
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref',default='6a7e87d')
    args=parser.parse_args();cv2.setNumThreads(config.INFERENCE_CV_THREADS)
    dest=ROOT.parent/'docs/stable-saber-89';dest.mkdir(parents=True,exist_ok=True)
    old=historical(args.baseline_ref)
    raw,mask,pose,face,keep=scene('down')
    rig=SuitRenderer();base=rig.render(raw.copy(),raw,mask,pose,face,keep,now=0)
    hands=lightsaber.glove_grips(rig.last_parts);tip=(600,387)
    tracker=lightsaber.SaberTracker();saber=tracker.update(tip,None,hands=hands,now=0,source=0)
    hand=lightsaber.glove_occlusion(rig.last_parts,tracker.hand_side,rig.assets['glove'],base.shape,saber)
    after=base.copy();lightsaber.draw_saber(after,saber,config.SABER_COLOR,occlusion=hand)
    cv2.imwrite(str(dest/'hand-integration.png'),after)
    if old:
        before=base.copy();old_saber=old.SaberTracker().update(tip,None,hands=hands)
        old.draw_saber(before,old_saber,(180,0,255))
        crop=lambda image:cv2.resize(image[335:530,505:700],(390,390))
        cv2.imwrite(str(dest/'contact-before-after.png'),np.hstack((label(crop(before),'Before: marker-colour glow'),label(crop(after),'After: cyan + current glove overlap'))))
    else:cv2.imwrite(str(dest/'contact-after.png'),cv2.resize(after[335:530,505:700],(390,390)))
    # Resting pen/glove jitter, isolated outlier, pose/marker dropouts and fast translation.
    stable=lightsaber.SaberTracker();legacy=old.SaberTracker() if old else None
    raw_tip=[];new_tip=[];old_grips=[];new_grips=[];times=[];cards=[];frames=[]
    for i in range(90):
        now=i/30
        shift=0 if i<60 else (i-60)*3
        noise=np.array((3 if i%2 else -3,2 if i%3 else -2))
        point=np.array((250+shift,105))+noise
        grip=np.array((260+shift,255))-noise
        current_hands={'r':tuple(grip)}
        if i==25:current_hands={'r':tuple(grip+(100,-80))}
        if 35<=i<=37:current_hands={}
        if 45<=i<=46:point=None
        start=time.perf_counter()
        weapon=stable.update(None if point is None else tuple(point),None,hands=current_hands,now=now,source=0)
        times.append((time.perf_counter()-start)*1000.)
        canvas=np.full((360,640,3),(39,31,24),np.uint8)
        cv2.ellipse(canvas,tuple(grip.astype(int)),(24,32),0,0,360,(105,145,185),-1,cv2.LINE_AA)
        hand_mask=np.zeros(canvas.shape[:2],np.uint8)
        cv2.ellipse(hand_mask,tuple(grip.astype(int)),(24,32),0,0,360,255,-1)
        lightsaber.draw_saber(canvas,weapon,config.SABER_COLOR,
                              occlusion=(0,0,hand_mask) if current_hands else None,opacity=stable.visibility)
        label(canvas,f't={now:.2f}s / visual fade={stable.visibility:.2f}')
        frames.append(canvas)
        if i in (10,25,36,46,50,75):cards.append(canvas)
        if point is not None and 5<=i<24:
            raw_tip.append(point.copy());new_tip.append(weapon.tip)
            new_grips.append((np.array(weapon.hilt_top)+weapon.hilt_bottom)/2)
        if legacy:
            prior=legacy.update(None if point is None else tuple(point),None,hands=current_hands)
            if point is not None and 5<=i<24:old_grips.append((np.array(prior.hilt_top)+prior.hilt_bottom)/2)
    cv2.imwrite(str(dest/'motion-cases.png'),np.vstack((np.hstack(cards[:3]),np.hstack(cards[3:]))))
    # Optional GIF export when Pillow is present; previews remain reproducible with OpenCV alone.
    try:
        from PIL import Image
        animation=[Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) for frame in frames]
        animation[0].save(dest/'motion-replay.gif',save_all=True,append_images=animation[1:],duration=33,loop=0)
    except ImportError:pass
    video=dest/'motion-replay.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30.,(640,360))
    if writer.isOpened():
        for frame in frames:writer.write(frame)
    writer.release()
    draws=[]
    for _ in range(60):
        image=base.copy();start=time.perf_counter()
        lightsaber.draw_saber(image,saber,config.SABER_COLOR,occlusion=hand)
        draws.append((time.perf_counter()-start)*1000.)
    def jitter(items):return round(float(np.std(items,axis=0).sum()),3)
    result=dict(camera=False,models=False,display=False,opencv_threads=cv2.getNumThreads(),
        static_tip_jitter_input_px=jitter(raw_tip),static_tip_jitter_output_px=jitter(new_tip),
        static_grip_jitter_before_px=jitter(old_grips) if old_grips else None,
        static_grip_jitter_after_px=jitter(new_grips),
        tracker_median_ms=round(statistics.median(times),3),
        draw_median_ms=round(statistics.median(draws),3),draw_p95_ms=round(float(np.percentile(draws,95)),3),
        note='Synthetic replay only. Jitter metric = sum of per-axis standard deviations during rest. No live FPS claim.')
    (dest/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
