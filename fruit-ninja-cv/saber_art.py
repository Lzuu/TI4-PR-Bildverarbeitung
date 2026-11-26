"""Cached laser-sword artwork: blade and grip fit independent tracked segments."""
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
        for name in ('blade','hilt'):
            part=image.copy()
            seam=int(meta['emitter'][1])
            if name=='blade':
                part[seam:,:,3]=0
                # Keep generated cyan spill bounded around the tracked blade;
                # the procedural halo below remains responsive to pen colour.
                distance=np.abs(np.arange(image.shape[1])-meta['tip'][0])
                support=np.clip((1.5-distance/meta['blade_width'])/.75,0,1)
                part[:,:,3]=np.rint(part[:,:,3]*support[None,:]).astype(np.uint8)
            else:part[:seam,:,3]=0
            anchors=np.float32([meta['tip'],meta['emitter']] if name=='blade' else
                               [meta['emitter'],meta['pommel']])
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
    for name,start,end,width in [('blade',saber.tip,saber.hilt_top,24),
                                 ('hilt',saber.hilt_top,saber.hilt_bottom,26)]:
        sprite,anchors,source_width=parts[name]
        matrix=part_matrix(anchors,start,end,source_width,max(3,width*scale))
        alpha_over(frame,sprite,matrix)
    return True
