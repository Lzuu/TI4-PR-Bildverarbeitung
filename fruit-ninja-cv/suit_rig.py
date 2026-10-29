"""Articulated EVA sprites, clear camera-face helmet and ROI alpha compositing.

All perception uses the caller's pristine BGR image. The suit is a layered
2.5D rig, not a stretch of one astronaut picture. No face is stored in assets.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import logging
from pathlib import Path
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)
ASSET_DIR = Path(__file__).resolve().parent / 'suit_assets'


@dataclass
class Sprite:
    image: np.ndarray
    premultiplied: np.ndarray
    bounds: tuple
    opening: np.ndarray = None

    @classmethod
    def prepare(cls, image):
        alpha = image[:, :, 3:4].astype(np.float32) / 255.
        rgba = np.concatenate((image[:, :, :3].astype(np.float32) * alpha, alpha), axis=2)
        return cls(image, rgba, cv2.boundingRect(image[:, :, 3]))


@lru_cache(maxsize=1)
def load_assets():
    """One bounded decoded set (8 × at most 384²), no growing scale cache."""
    meta = json.loads((ASSET_DIR / 'anchors.json').read_text(encoding='utf-8'))['parts']
    assets = {}
    for name in meta:
        image = cv2.imread(str(ASSET_DIR / (name + '.png')), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise RuntimeError('EVA sprite missing or without alpha: ' + name)
        factor = min(1., 384 / max(image.shape[:2]))
        if factor < 1:
            image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        assets[name] = Sprite.prepare(image)
        if 'width' in meta[name]:
            # Generated artwork need not obey the requested margin exactly.
            # Fit actual opaque garment width, not transparent canvas padding.
            meta[name]['width']=assets[name].bounds[2]/image.shape[1]
        if name=='helmet':
            transparent=(image[:,:,3]<20).astype(np.uint8)
            _,labels=cv2.connectedComponents(transparent)
            component=labels[image.shape[0]//2,image.shape[1]//2]
            if component==0 or component==labels[0,0]:
                raise RuntimeError('Helmet needs a genuinely transparent enclosed aperture')
            hole=(labels==component).astype(np.uint8)
            assets[name].opening=cv2.dilate(hole,np.ones((3,3),np.uint8)).astype(np.float32)
            x,y,bw,bh=cv2.boundingRect(hole)
            meta[name]['opening_center']=[(x+bw/2)/image.shape[1],(y+bh/2)/image.shape[0]]
            meta[name]['opening_axes']=[bw/2/image.shape[1],bh/2/image.shape[0]]
    return assets, meta


def segment_transform(anchors, start, end, width, source_width):
    """Anisotropic affine map: garment axis follows bone, width is perpendicular."""
    a, b = np.asarray(anchors, np.float32)
    start, end = np.asarray(start, np.float32), np.asarray(end, np.float32)
    delta = end - start
    length = float(np.linalg.norm(delta))
    original_length = float(np.linalg.norm(b - a))
    if length < 1 or original_length < 1 or source_width < 1:
        return None
    direction = delta / length
    normal = np.array((direction[1], -direction[0]), np.float32)
    # The authoring convention has a vertical proximal-to-distal centre line.
    original_direction = (b - a) / original_length
    original_normal = np.array((original_direction[1], -original_direction[0]))
    linear = (np.outer(direction, original_direction) * (length / original_length)
              + np.outer(normal, original_normal) * (width / source_width))
    return np.column_stack((linear, start - linear @ a)).astype(np.float32)


def _warp_roi(data, matrix, shape, bounds=None):
    """Warp only the clipped destination rectangle, for affine or perspective."""
    if matrix is None or not np.isfinite(matrix).all():
        return None
    h, w = data.shape[:2]
    x, y, bw, bh = bounds or (0, 0, w, h)
    corners = np.float32([[[x, y], [x+bw, y], [x+bw, y+bh], [x, y+bh]]])
    if matrix.shape == (3, 3):
        dest = cv2.perspectiveTransform(corners, matrix)[0]
    else:
        dest = cv2.transform(corners, matrix)[0]
    x0, y0 = np.maximum(np.floor(dest.min(axis=0)) - 1, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(dest.max(axis=0)) + 2, (shape[1], shape[0])).astype(int)
    if x1 <= x0 or y1 <= y0:
        return None
    offset = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float32)
    if matrix.shape == (3, 3):
        local = offset @ matrix
        warped = cv2.warpPerspective(data, local, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR)
    else:
        local = matrix.copy()
        local[:, 2] -= (x0, y0)
        warped = cv2.warpAffine(data, local, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR)
    return x0, y0, x1, y1, warped


def alpha_over(frame, sprite, matrix, mask=None):
    """Premultiplied interpolation avoids black fringes; writes only a local ROI."""
    prepared = sprite if isinstance(sprite, Sprite) else Sprite.prepare(sprite)
    result = _warp_roi(prepared.premultiplied, matrix, frame.shape, prepared.bounds)
    if result is None:
        return frame
    x0, y0, x1, y1, warped = result
    alpha = warped[:, :, 3:4]
    color = warped[:, :, :3]
    if mask is not None:
        factor = mask[y0:y1, x0:x1, None].astype(np.float32) / 255.
        alpha = alpha * factor
        color = color * factor
    roi = frame[y0:y1, x0:x1]
    roi[:] = np.clip(color + roi * (1 - alpha), 0, 255).astype(np.uint8)
    return frame


def _component(mask):
    """Largest player, preserving real holes (no filling between crossed arms)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < mask.size * .01:
        return None, None
    selection = np.zeros_like(mask)
    cv2.drawContours(selection, [contour], -1, 255, -1)
    return cv2.bitwise_and(mask, selection), cv2.boundingRect(contour)


def face_from_pose(pose):
    if pose is None or not all(pose.valid(i) for i in (2, 5, 0)):
        return None
    eyes = pose.points[[2, 5], :2]
    distance = float(np.linalg.norm(eyes[1] - eyes[0]))
    if distance < 3:
        return None
    cx, cy = eyes.mean(axis=0)
    return (float(cx-distance*1.3), float(cy-distance*.8), distance*2.6, distance*3.4)


def _roll(pose):
    if pose is None or not all(pose.valid(i) for i in (2, 5)):
        return 0.
    a, b = sorted(pose.points[[2, 5], :2], key=lambda p: p[0])
    return float(np.clip(np.degrees(np.arctan2(*(b-a)[::-1])), -40, 40))


class SuitRenderer:
    def __init__(self):
        self.assets, self.meta = load_assets()
        self.head = None
        self._head_time = None
        self._source = None
        self._shape = None
        self.last_parts = []
        self.last_ms = 0.
        self._widths={}
        self._face_calibration=None

    def reset(self):
        self.head = None
        self._head_time = None
        self.last_parts = []
        self._widths={}
        self._face_calibration=None

    def _head(self, face, pose, now):
        measured = face if face is not None else face_from_pose(pose)
        eye_points=(pose.points[[2,5],:2] if pose is not None
                    and all(pose.valid(i) for i in (2,5)) else None)
        if eye_points is not None:
            distance=float(np.linalg.norm(eye_points[1]-eye_points[0]))
            center=eye_points.mean(axis=0)
            if distance>=3 and face is not None:
                x,y,w,h=face
                self._face_calibration=(w/distance,h/distance,
                                        (np.array((x+w/2,y+h/2))-center)/distance)
            elif distance>=3 and self._face_calibration is not None:
                wr,hr,offset=self._face_calibration
                w,h=distance*wr,distance*hr
                cx,cy=center+offset*distance
                measured=(cx-w/2,cy-h/2,w,h)
        if measured is not None:
            box = np.asarray(measured, np.float32)
            if np.isfinite(box).all() and min(box[2:]) >= 5:
                # Keep current position in camera coordinates; smooth only small size noise.
                if self.head is not None:
                    previous = np.asarray(self.head[0])
                    relative = np.max(np.abs(box[2:] / previous[2:] - 1))
                    if relative < .18:
                        center = box[:2] + box[2:] / 2
                        box[2:] = previous[2:] * .6 + box[2:] * .4
                        box[:2] = center - box[2:] / 2
                self.head = (tuple(box), _roll(pose))
                self._head_time = now
        elif self._head_time is not None and now - self._head_time > .12:
            self.head = None
        return self.head

    def _limb(self, asset, label, a, b, width, pose, mask=None, torso=None):
        endpoints = pose.segment(a, b)
        if endpoints is None or np.linalg.norm(endpoints[1]-endpoints[0]) < 3:
            return None
        if mask is not None:
            midpoint=endpoints.mean(axis=0)
            inside=(torso is not None and cv2.pointPolygonTest(torso[[0,1,3,2]],tuple(map(float,midpoint)),False)>=0)
            if not inside or asset=='upper_arm':
                delta=endpoints[1]-endpoints[0]
                normal=np.array((delta[1],-delta[0]))/max(np.linalg.norm(delta),1)
                span=np.linspace(-width*.95,width*.95,65)
                sizes=[]
                for fraction in (.3,.5,.7):
                    center=endpoints[0]+delta*fraction
                    samples=np.rint(center+span[:,None]*normal).astype(int)
                    valid=(samples[:,0]>=0)&(samples[:,0]<mask.shape[1])&(samples[:,1]>=0)&(samples[:,1]<mask.shape[0])
                    hits=np.zeros(65,bool)
                    hits[valid]=mask[samples[valid,1],samples[valid,0]]>0
                    if not hits[32]:continue
                    left=right=32
                    while left>0 and hits[left-1]:left-=1
                    while right<64 and hits[right+1]:right+=1
                    sizes.append(span[right]-span[left])
                if sizes:
                    measured=float(np.clip(np.percentile(sizes,35)*1.06,width*.8,width*1.75))
                    self._widths[label]=measured/width
            width*=self._widths.get(label,1.)
        return self._part(asset, label, endpoints, width)

    def _part(self, asset, label, endpoints, width):
        sprite, meta = self.assets[asset], self.meta[asset]
        h, w = sprite.image.shape[:2]
        anchors = np.asarray(meta['anchors'], np.float32) * (w, h)
        matrix = segment_transform(anchors, *endpoints, width, meta['width'] * w)
        return dict(name=label, asset=asset, matrix=matrix, endpoints=np.asarray(endpoints).copy())

    def _torso(self, pose, bbox, face):
        if pose is not None and all(pose.valid(i) for i in (11, 12)):
            shoulders = pose.points[[11, 12], :2].copy()
            shoulders = shoulders[np.argsort(shoulders[:, 0])]
            if all(pose.valid(i) for i in (23, 24)):
                hips = pose.points[[23, 24], :2].copy()
                hips = hips[np.argsort(hips[:, 0])]
            else:
                # Torso continuation only: a cropped waist is not an invented arm/leg.
                down = np.array((0, np.linalg.norm(shoulders[1]-shoulders[0])*1.45))
                hips = shoulders + down
            span=shoulders[1]-shoulders[0]
            shoulders[0]-=span*.10;shoulders[1]+=span*.10
            hips[0]-=span*.13;hips[1]+=span*.13
            return np.float32([shoulders[0], shoulders[1], hips[0], hips[1]])
        if face is not None:
            x, y, w, h = face
            cx = x + w/2
            return np.float32([[cx-w*1.65,y+h*1.05],[cx+w*1.65,y+h*1.05],
                               [cx-w*1.2,y+h*4.3],[cx+w*1.2,y+h*4.3]])
        return None

    def _helmet(self, frame, original, head, phase=0, person_mask=None):
        (fx, fy, fw, fh), angle = head
        if fw < 5 or fh < 5:
            return
        sprite = self.assets['helmet']
        h, w = sprite.image.shape[:2]
        # The locally generated shell's clear aperture surrounds the unwarped face.
        opening = self.meta['helmet']
        center = np.asarray(opening['opening_center']) * (w, h)
        axes = np.asarray(opening['opening_axes']) * (w, h)
        sx, sy = fw * .66 / axes[0], fh * .48 / axes[1]
        radians = np.radians(angle)
        rotation = np.float32([[np.cos(radians),-np.sin(radians)],
                               [np.sin(radians),np.cos(radians)]])
        linear = rotation @ np.diag((sx,sy))
        target = np.array((fx+fw/2,fy+fh*.5))
        matrix = np.column_stack((linear, target-linear@center)).astype(np.float32)
        # Paint camera pixels into a bounded aperture BEFORE shell and glass.
        aperture=sprite.opening
        warped = _warp_roi(aperture,matrix,frame.shape)
        if warped is not None:
            x0,y0,x1,y1,coverage=warped
            roi, raw = frame[y0:y1,x0:x1],original[y0:y1,x0:x1]
            coverage=np.clip(coverage,0,1)
            if person_mask is None:
                camera=raw
            else:
                support=person_mask[y0:y1,x0:x1].astype(np.float32)/255
                # Keep actual face pixels even if the coarse person model has a hole.
                face_support=np.zeros(support.shape,np.uint8)
                cv2.ellipse(face_support,(round(fx+fw/2-x0),round(fy+fh/2-y0)),
                            (max(1,round(fw*.5)),max(1,round(fh*.53))),angle,0,360,255,-1,cv2.LINE_AA)
                support=np.maximum(support,face_support/255.)
                interior=np.array((31,34,38),np.float32)
                camera=raw*support[:,:,None]+interior*(1-support[:,:,None])
            roi[:]=(camera*coverage[:,:,None]+roi*(1-coverage[:,:,None])).astype(np.uint8)
        alpha_over(frame,sprite,matrix)
        if warped is not None:
            # Glass is near-clear centrally; reflections stay at the lateral edge.
            yy,xx=np.mgrid[y0:y1,x0:x1]
            coords=np.stack((xx-target[0],yy-target[1]),axis=-1)@rotation
            nx,ny=coords[:,:,0]/max(fw*.66,1),coords[:,:,1]/max(fh*.48,1)
            edge=np.clip((nx*nx+ny*ny-.45)/.55,0,1)
            glint=np.exp(-((nx+.77+.025*np.sin(phase*.35))/.065)**2)*.07
            strength=(.018*edge+glint)*coverage
            tint=np.array((220,225,230),np.float32)
            roi[:]=np.clip(roi*(1-strength[:,:,None])+tint*strength[:,:,None],0,255).astype(np.uint8)

    def _fit_waist(self,quad,mask):
        """Skeletal hips are internal joints; the garment fits the outer body."""
        center=(quad[2]+quad[3])/2
        y,x=round(float(center[1])),round(float(center[0]))
        if not (0<=y<mask.shape[0] and 0<=x<mask.shape[1] and mask[y,x]):
            return quad
        left=right=x
        while left>0 and mask[y,left-1]:left-=1
        while right<mask.shape[1]-1 and mask[y,right+1]:right+=1
        shoulder=float(np.linalg.norm(quad[1]-quad[0]))
        hip=quad[3]-quad[2]
        length=float(np.linalg.norm(hip))
        if length<2:return quad
        target=float(np.clip((right-left)*.85,length*.85,shoulder*2.1))
        fitted=quad.copy()
        fitted[2]=center-hip/length*target/2
        fitted[3]=center+hip/length*target/2
        return fitted

    def render(self, frame, original, mask, pose=None, face=None, keep=None,
               now=None, source=None):
        started=time.perf_counter()
        now=time.monotonic() if now is None else now
        if np.shares_memory(frame,original):
            original=original.copy()
        if source != self._source or frame.shape != self._shape:
            self.reset()
        self._source,self._shape=source,frame.shape
        player,bbox=_component(mask)
        self.last_parts=[]
        if bbox is None:
            self.reset()
            return frame
        head=self._head(face,pose,now)
        face_box=head[0] if head is not None else None
        if pose is None and head is None:
            # No measured geometry: show the real person, never white out a face
            # or leave a guessed helmet/limb at the old position.
            cv2.copyTo(original,player,frame)
            if keep is not None:cv2.copyTo(original,keep,frame)
            self.last_ms=(time.perf_counter()-started)*1000
            return frame
        # Mask only inward to soften edges without painting the background.
        distance=cv2.distanceTransform(player,cv2.DIST_L2,3)
        clip=np.minimum(distance/1.4,1)*255
        clip=clip.astype(np.uint8)
        region=player.copy()
        if face_box is not None:
            x,y,w,h=face_box
            cv2.ellipse(region,(round(x+w/2),round(y+h*.45)),
                        (max(1,round(w*.7)),max(1,round(h*.75))),0,0,360,0,-1)
        elif pose is not None and all(pose.valid(i) for i in (11,12)):
            # A body pose alone is not permission to recolour an unlocated face.
            # Measured arm sprites can still render above this conservative line.
            shoulder_y=int(min(pose.points[11,1],pose.points[12,1]))
            region[:max(0,shoulder_y)]=0
        if keep is not None:
            region[keep>0]=0
        # Base fabric fills small joints between the articulated pieces, never the face.
        import suit
        base=frame.copy()
        suit.paint_fabric(base,region)
        cv2.copyTo(base,region,frame)
        width=0
        if pose is not None and pose.segment(11,12) is not None:
            width=float(np.linalg.norm(pose.points[11,:2]-pose.points[12,:2]))
        commands=[]
        quad=self._torso(pose,bbox,face_box)
        if quad is not None and pose is not None:
            quad=self._fit_waist(quad,player)
        if width>=8:
            for side,indices in [('l',(11,13,15,19,23,25,27,31)),
                                 ('r',(12,14,16,20,24,26,28,32))]:
                sh,el,wr,fi,hip,knee,ankle,toe=indices
                for asset,a,b,factor,layer in [('thigh',hip,knee,.42,0),('shin',knee,ankle,.34,1),
                                              ('upper_arm',sh,el,.42,20),('forearm',el,wr,.29,40)]:
                    part=self._limb(asset,asset+'_'+side,a,b,width*factor,pose,player,quad)
                    if part is not None:
                        # Far arms go behind the torso; near forearms cross in front.
                        if 'arm' in asset and np.mean(pose.points[[a,b],2])>width*.12:
                            layer=5
                        commands.append((layer,part))
                hand=pose.segment(wr,fi)
                if hand is None and pose.segment(el,wr) is not None:
                    a,b=pose.segment(el,wr)
                    direction=(b-a)/max(np.linalg.norm(b-a),1)
                    hand=np.array((b,b+direction*width*.27))
                if hand is not None:
                    commands.append((45,self._part('glove','glove_'+side,hand,width*.26)))
                foot=pose.segment(ankle,toe)
                if foot is not None:
                    if np.linalg.norm(foot[1]-foot[0])<width*.17:
                        foot[1]=foot[0]+(0,width*.25)
                    commands.append((2,self._part('boot','boot_'+side,foot,width*.34)))
        if quad is not None:
            sprite=self.assets['torso']
            source_quad=np.asarray(self.meta['torso']['quad'],np.float32)*(sprite.image.shape[1],sprite.image.shape[0])
            # getPerspectiveTransform wants perimeter order, not two rows.
            matrix=cv2.getPerspectiveTransform(source_quad[[0,1,3,2]].astype(np.float32),quad[[0,1,3,2]])
            commands.append((10,dict(name='torso',asset='torso',matrix=matrix,endpoints=quad)))
        # Keep upper-arm/torso below the helmet, wrists/gloves can occlude it.
        for layer,part in sorted(commands,key=lambda item:item[0]):
            if layer>=40:
                continue
            alpha_over(frame,self.assets[part['asset']],part['matrix'],clip)
            self.last_parts.append(part)
        if head is not None:
            self._helmet(frame,original,head,now,player)
        for layer,part in sorted(commands,key=lambda item:item[0]):
            if layer<40:
                continue
            alpha_over(frame,self.assets[part['asset']],part['matrix'],clip)
            self.last_parts.append(part)
        if keep is not None:
            cv2.copyTo(original,keep,frame)  # includes pen excluded by the person model
        self.last_ms=(time.perf_counter()-started)*1000
        return frame
