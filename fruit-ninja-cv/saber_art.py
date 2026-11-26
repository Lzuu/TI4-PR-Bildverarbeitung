"""Material-only grip from the generated sword; energy is rendered in lightsaber.

The original PNG is kept intact. Its baked energy halo is deliberately not used:
it becomes a coloured rectangle when stretched to a short tracked blade.
"""
from functools import lru_cache
import json
import logging
from pathlib import Path
import numpy as np
import cv2
from utils import imread
from suit_rig import Sprite, alpha_over, segment_transform

logger=logging.getLogger(__name__)
ASSET_DIR=Path(__file__).resolve().parent/'weapon_assets'


@lru_cache(maxsize=1)
def load_parts():
    try:
        meta=json.loads((ASSET_DIR/'laser-sword.json').read_text(encoding='utf-8'))
        image=imread(ASSET_DIR/'laser-sword.png',cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim!=3 or image.shape[2]!=4:
            raise ValueError('Missing RGBA laser sword')
        result={}
        for name in ('hilt',):
            part=image.copy()
            seam=int(meta['emitter'][1])
            part[:seam,:,3]=0
            # Keep material edges, removing the low-alpha ambient wash.
            alpha=np.clip((part[:,:,3].astype(np.float32)-64)/191,0,1)
            distance=np.abs(np.arange(image.shape[1])-meta['emitter'][0])
            support=np.clip((.8-distance/meta['hilt_width'])/.15,0,1)
            part[:,:,3]=np.rint(alpha*support[None,:]*255).astype(np.uint8)
            anchors=np.float32([meta['emitter'],meta['pommel']])
            result[name]=(Sprite.prepare(part),anchors,float(meta[name+'_width']))
        return result
    except (OSError,ValueError,KeyError,TypeError):
        logger.warning('Laser sword artwork unavailable; using procedural weapon',exc_info=True)
        return None


def part_matrix(anchors,start,end,source_width,target_width):
    return segment_transform(anchors,start,end,target_width,source_width)


def draw(frame,saber,scale):
    parts=load_parts()
    if parts is None:
        return False
    for name,start,end,width in [('hilt',saber.hilt_top,saber.hilt_bottom,20)]:
        sprite,anchors,source_width=parts[name]
        matrix=part_matrix(anchors,start,end,source_width,max(3,width*scale))
        alpha_over(frame,sprite,matrix)
    return True
