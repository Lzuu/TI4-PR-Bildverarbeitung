"""Local generated artwork, bounded size cache and small-ROI alpha compositing.

The 320px authoring canvas places the collision-body centre at (160,184)
with radius 120. The bomb uses a 400px canvas centred at (200,260) to fit the
wick without shrinking its collision body. Stems/fuse are decorative.
"""
from functools import lru_cache
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from utils import imread

logger=logging.getLogger(__name__)
ASSET_DIR=Path(__file__).resolve().parent/'fruit_assets'
NAMES=('apple','orange','lime','blueberry','lemon','bomb')


@lru_cache(maxsize=6)
def _original(name):
    if name not in NAMES:
        return None
    image=imread(ASSET_DIR/(name+'.png'),cv2.IMREAD_UNCHANGED)
    canvas=400 if name=='bomb' else 320
    if image is None or image.shape!=(canvas,canvas,4):
        logger.warning('Fruit artwork unavailable: %s; using procedural fallback',name)
        return None
    alpha=image[:,:,3:4].astype(np.float32)/255
    return np.concatenate((image[:,:,:3].astype(np.float32)*alpha,alpha),axis=2)


@lru_cache(maxsize=128)
def get_sprite(name,radius):
    """Size artwork once, interpolating premultiplied colour to avoid dark rims."""
    source=_original(name)
    if source is None:
        return None
    size=max(3,min(2048,round(source.shape[0]*radius/120)))
    premult=cv2.resize(source,(size,size),interpolation=cv2.INTER_AREA if size<320 else cv2.INTER_LINEAR)
    alpha=premult[:,:,3:4]
    colour=np.divide(premult[:,:,:3],alpha,out=np.zeros_like(premult[:,:,:3]),where=alpha>1e-6)
    image=np.rint(np.clip(np.concatenate((colour,alpha*255),axis=2),0,255)).astype(np.uint8)
    image.flags.writeable=False
    return image


def _over(frame,image,center,selection=None,anchor_y=184/320):
    h,w=image.shape[:2]
    left=round(center[0]-w*.5);top=round(center[1]-h*anchor_y)
    x0,y0=max(0,left),max(0,top)
    x1,y1=min(frame.shape[1],left+w),min(frame.shape[0],top+h)
    if x1<=x0 or y1<=y0:
        return
    sx,sy=x0-left,y0-top
    sprite=image[sy:sy+y1-y0,sx:sx+x1-x0]
    alpha=sprite[:,:,3:4].astype(np.float32)/255
    if selection is not None:
        alpha*=selection[sy:sy+y1-y0,sx:sx+x1-x0,None]
    roi=frame[y0:y1,x0:x1]
    roi[:]=np.clip(sprite[:,:,:3]*alpha+roi*(1-alpha),0,255).astype(np.uint8)


def draw(frame,name,center,radius,cut_angle=None,spread=0):
    """Return False for a missing asset so the caller can keep playing."""
    image=get_sprite(name,radius)
    if image is None:
        return False
    if cut_angle is None or name=='bomb':
        _over(frame,image,center,anchor_y=260/400 if name=='bomb' else 184/320)
    else:
        h,w=image.shape[:2]
        yy,xx=np.ogrid[:h,:w]
        nx,ny=-math.sin(cut_angle),math.cos(cut_angle)
        positive=(xx-w*.5)*nx+(yy-h*184/320)*ny>=0
        _over(frame,image,(center[0]+nx*spread,center[1]+ny*spread),positive)
        _over(frame,image,(center[0]-nx*spread,center[1]-ny*spread),~positive)
    return True
