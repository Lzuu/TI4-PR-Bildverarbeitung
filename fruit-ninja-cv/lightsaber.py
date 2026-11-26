"""Time-stabilized visual laser sword with a material grip and cyan light.

The drawn tip damps small jitter within a 3px*scale error bound; fast movement
passes through. The grip follows the current pose glove with a short outlier
gate and sticky hand assignment. Skin search is a fallback after a brief pose
grace period. Short marker dropouts fade visually; no held point enters gameplay.
Real frame timestamps and source identity prevent frame-rate-dependent holds
or carrying a weapon across camera/mode changes. Current hand pixels overlap the
grip; an analytic halo avoids stretching the generated image's coloured wash.
"""

import logging
import math
from collections import namedtuple

import cv2
import numpy as np

import config
import saber_art

logger = logging.getLogger(__name__)

Saber = namedtuple("Saber", "tip hilt_top hilt_bottom")


def glove_grips(parts, along=None):
    """Grip points of the gloves the suit drew: {side: (x, y)}, e.g. {"l": ..., "r": ...}.

    ``parts`` is ``SuitRenderer.last_parts`` of the current frame; a glove's
    endpoints run from the wrist to the fingertips, the grip is the palm centre
    ``along`` (config.SABER_GLOVE_GRIP) of the way.
    """
    along = config.SABER_GLOVE_GRIP if along is None else along
    grips = {}
    for part in parts or ():
        name = part.get("name", "")
        if name.startswith("glove_"):
            a, b = np.asarray(part["endpoints"], np.float32)[:2]
            grip = a + along * (b - a)
            grips[name[len("glove_"):]] = (float(grip[0]), float(grip[1]))
    return grips


_SKIN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def skin_mask(frame):
    """Skin-coloured pixels (YCrCb range), used to find the hand holding the pen.

    Computed at half resolution (4x fewer pixels, the mask is coarse anyway) on
    the raw camera image -- so the hand is found even when the suit shows it as
    a glove. Clothes in skin tones count as skin too (known limitation).
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    skin = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb), (0, 133, 77), (255, 173, 127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, _SKIN_KERNEL)
    return cv2.resize(cv2.dilate(skin, _SKIN_KERNEL), (w, h), interpolation=cv2.INTER_NEAREST)


def find_hand(skin, tip, radius, exclude=None, min_area=None, hand_size=None):
    """Where the hand holding the pen is, or None.

    The nearest skin blob (by its closest pixel) within ``radius`` of ``tip``;
    blobs centred in ``exclude`` (the face box x, y, w, h) are ignored. The
    result is the centre of the part of that blob next to the pen (within
    ``hand_size`` of its closest pixel) -- a hand touching a bare arm or a
    skin-coloured object must not drift into the middle of it all.
    """
    if skin is None:
        return None
    if min_area is None:
        min_area = 400 * config.SCALE ** 2
    if hand_size is None:
        hand_size = 80 * config.SCALE
    h, w = skin.shape[:2]
    x0, y0 = max(int(tip[0] - radius), 0), max(int(tip[1] - radius), 0)
    x1, y1 = min(int(tip[0] + radius) + 1, w), min(int(tip[1] + radius) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return None
    window = cv2.threshold(skin[y0:y1, x0:x1], 127, 255, cv2.THRESH_BINARY)[1]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
    if n <= 1:
        return None
    # distances only for the skin pixels (far fewer than the whole window)
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    xs = xs.astype(np.float32) + x0
    ys = ys.astype(np.float32) + y0
    dist = np.hypot(xs - tip[0], ys - tip[1])
    nearest = np.full(n, np.inf, np.float32)
    np.minimum.at(nearest, lab, dist)                # closest pixel of every blob
    best, best_dist = None, None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area or nearest[i] > radius:
            continue
        if exclude is not None:
            cx, cy = centroids[i][0] + x0, centroids[i][1] + y0
            ex, ey, ew, eh = exclude
            if ex - 0.3 * ew <= cx <= ex + 1.3 * ew and ey - 0.3 * eh <= cy <= ey + 1.3 * eh:
                continue
        if best_dist is None or nearest[i] < best_dist:
            best, best_dist = i, nearest[i]
    if best is None:
        return None
    part = (lab == best) & (dist <= best_dist + hand_size)
    return float(xs[part].mean()), float(ys[part].mean())


def saber_toward(tip, direction, blade_length=None, hilt_length=None):
    """Saber of fixed length from ``tip`` along ``direction`` (unit vector, blade then hilt)."""
    if blade_length is None:
        blade_length = config.SABER_LENGTH * config.SCALE
    if hilt_length is None:
        hilt_length = config.SABER_HILT_LENGTH * config.SCALE
    dx, dy = direction
    as_int = lambda q: (int(round(q[0])), int(round(q[1])))
    top = (tip[0] + dx * blade_length, tip[1] + dy * blade_length)
    bottom = (top[0] + dx * hilt_length, top[1] + dy * hilt_length)
    return Saber((int(tip[0]), int(tip[1])), as_int(top), as_int(bottom))


MAX_TILT = math.radians(75)      # largest tilt from straight down


class SaberTracker:
    """Stable visual weapon; gameplay continues using unfiltered live pen samples.

    Explicit timestamps are used in the game. Timestamp-free callers advance at
    REF_FPS for deterministic standalone use. Only a short visual dropout is held.
    """
    def __init__(self, smoothing=0.6, blade_length=None, hilt_length=None, radius=None,
                 confirm_frames=4, jump_angle=45, keep_frames=None,
                 adapt_rate=0.05, min_blade=None, max_blade=None):
        self.smoothing = smoothing
        self.blade_length, self.hilt_length, self.radius = blade_length, hilt_length, radius
        self.confirm_frames, self.jump = confirm_frames, math.radians(jump_angle)
        self.grip_hold = config.SABER_GRIP_HOLD if keep_frames is None else keep_frames/config.REF_FPS
        self.adapt_rate, self.min_blade, self.max_blade = adapt_rate, min_blade, max_blade
        self._source = None
        self._streak = 0
        self._reset()

    def _reset(self):
        self.angle, self.length = 0., None
        self._pending = 0.
        self._side = self._switch = None
        self._tracking_source = None
        self._last_time = self._tip_time = self._grip_time = None
        self._tip = self._saved = self._offset = self._grip_tip = None
        self._candidate = None
        self._angle_candidate = None
        self.visibility = 0.

    @property
    def hand_side(self):
        return self._side

    def _radius(self):
        return self.radius if self.radius is not None else config.SABER_HAND_RADIUS*config.SCALE

    def _min_blade(self):
        return self.min_blade if self.min_blade is not None else 60*config.SCALE

    def _max_blade(self):
        return self.max_blade if self.max_blade is not None else 320*config.SCALE

    def _adapt_length(self, distance, hilt):
        wanted = min(max(distance-hilt/2, self._min_blade()), self._max_blade())
        alpha = 1-(1-self.adapt_rate)**(self._dt*config.REF_FPS)
        self.length += (wanted-self.length)*alpha

    def _target(self, tip, skin, face):
        hand = find_hand(skin, tip, self._radius(), face)
        if hand is None:
            return None
        dx, dy = hand[0]-tip[0], hand[1]-tip[1]
        if dy <= 0:
            return None
        angle = math.atan2(dx, dy)
        return (angle, math.hypot(dx, dy)) if abs(angle)<=MAX_TILT else None

    def _glove(self, tip, hands):
        reach = {side:math.dist(tip,grip) for side,grip in (hands or {}).items()
                 if np.isfinite(grip).all() and math.dist(tip,grip)<=self._radius()}
        if not reach:
            self._switch = None
            return None
        best = min(reach,key=reach.get)
        if self._side is None:
            self._side = best
        elif best != self._side:
            clearly_closer = self._side not in reach or reach[best]<config.SABER_HAND_SWITCH*reach[self._side]
            if clearly_closer:
                if self._switch is None or self._switch[0]!=best:
                    self._switch = (best,self._now)
                if self._now-self._switch[1]>=config.SABER_HAND_SWITCH_SECONDS:
                    self._side,self._switch,self._offset = best,None,None
            else:
                self._switch = None
        else:
            self._switch = None
        return hands[self._side] if self._side in reach else None

    @staticmethod
    def _translate(saber, tip):
        delta=np.asarray(tip)-saber.tip
        point=lambda value:tuple(np.rint(np.asarray(value)+delta).astype(int))
        return Saber(tuple(np.rint(tip).astype(int)),point(saber.hilt_top),point(saber.hilt_bottom))

    def _on_glove(self, tip, grip, hilt):
        tip=np.asarray(tip,float);desired=np.asarray(grip,float)-tip
        if self._offset is not None and self._grip_time is not None and self._now-self._grip_time>.35:
            self._offset=None
        if self._offset is not None:
            difference=float(np.linalg.norm(desired-self._offset))
            translation=0. if self._grip_tip is None else float(np.linalg.norm(tip-self._grip_tip))
            if difference>70*config.SCALE+2*translation:
                if self._candidate is None or np.linalg.norm(desired-self._candidate[0])>30*config.SCALE:
                    self._candidate=(desired.copy(),self._now)
                if self._now-self._candidate[1]<.05:
                    return self._translate(self._saved,tip)
            else:
                self._candidate=None
            alpha=1-math.exp(-self._dt/.055)
            fitted=self._offset+(desired-self._offset)*alpha
            error=desired-fitted;distance=float(np.linalg.norm(error))
            if distance>5*config.SCALE:
                fitted=desired-error/distance*5*config.SCALE
            self._offset=fitted
        else:
            self._offset=desired
        self._candidate=None
        self._grip_time,self._grip_tip=self._now,tip.copy()
        grip=tip+self._offset
        dx,dy=grip-tip;distance=math.hypot(dx,dy)
        # An end-on pen provides no reliable axis; retain the previous one.
        if distance>=max(12*config.SCALE,hilt*.4):
            self.angle=math.atan2(dx,dy)
        self.length=min(max(distance-hilt/2,self._min_blade()),self._max_blade())
        self._pending=0.
        direction=np.array((math.sin(self.angle),math.cos(self.angle)))
        point=lambda value:tuple(np.rint(value).astype(int))
        return Saber(point(tip),point(grip-direction*hilt/2),point(grip+direction*hilt/2))

    def _note(self, source):
        if source==self._source:
            self._streak=0.
            return
        self._streak+=self._dt*config.REF_FPS
        if self._streak+1e-6>=self.confirm_frames:
            self._source,self._streak=source,0.
            logger.info('Saber hilt %s','on the suit glove' if source=='glove' else 'on the skin hand')

    def update(self, tip, skin, face=None, hands=None, now=None, source=None):
        now=(0. if self._last_time is None else self._last_time+1/config.REF_FPS) if now is None else float(now)
        if not math.isfinite(now):
            self._reset()
            return None
        if source!=self._tracking_source or (self._last_time is not None and now<self._last_time):
            self._reset()
        self._tracking_source=source
        self._dt=1/config.REF_FPS if self._last_time is None else max(0.,min(.1,now-self._last_time))
        self._now=self._last_time=now
        if tip is None or not np.isfinite(tip).all():
            age=float('inf') if self._tip_time is None else now-self._tip_time
            self.visibility=max(0.,1-age/config.SABER_MARKER_HOLD)
            if self._saved is not None and age<config.SABER_MARKER_HOLD:
                return self._saved
            self._reset()
            self._tracking_source=source;self._last_time=now
            return None
        tip=np.asarray(tip,float)
        if self._tip is not None:
            delta=tip-self._tip;distance=float(np.linalg.norm(delta))
            if distance<12*config.SCALE:
                fitted=self._tip+delta*(1-math.exp(-self._dt/.055))
                error=tip-fitted;length=float(np.linalg.norm(error))
                self._tip=fitted if length<=3*config.SCALE else tip-error/length*3*config.SCALE
            else:self._tip=tip
        else:self._tip=tip
        tip=tuple(np.rint(self._tip).astype(int))
        self._tip_time,self.visibility=now,1.
        hilt=self.hilt_length if self.hilt_length is not None else config.SABER_HILT_LENGTH*config.SCALE
        if self.length is None:
            self.length=self.blade_length if self.blade_length is not None else config.SABER_LENGTH*config.SCALE
        grip=self._glove(tip,hands)
        if grip is not None:
            self._note('glove')
            self._saved=self._on_glove(tip,grip,hilt)
            return self._saved
        if self._saved is not None and self._grip_time is not None and now-self._grip_time<=self.grip_hold:
            self._saved=self._translate(self._saved,tip)
            return self._saved
        self._note('skin')
        if callable(skin):skin=skin()
        found=self._target(tip,skin,face)
        if found is not None:
            difference=(found[0]-self.angle+math.pi)%(2*math.pi)-math.pi
            if abs(difference)>self.jump:
                if self._angle_candidate is None or abs((found[0]-self._angle_candidate+math.pi)%(2*math.pi)-math.pi)>math.radians(15):
                    self._pending=0.
                    self._angle_candidate=found[0]
                self._pending+=self._dt*config.REF_FPS
                if self._pending+1e-6>=self.confirm_frames:
                    self.angle,self._pending=found[0],0.
                    self._angle_candidate=None
                    self._adapt_length(found[1],hilt)
            else:
                self._pending=0.
                self._angle_candidate=None
                self.angle+=difference*(1-self.smoothing**(self._dt*config.REF_FPS))
                self._adapt_length(found[1],hilt)
        else:
            self._pending=0.  # Missing hand must not make the weapon rotate by itself.
            self._angle_candidate=None
        self._saved=saber_toward(tip,(math.sin(self.angle),math.cos(self.angle)),self.length,hilt)
        return self._saved


def _add_glow(frame, a, b, color, thickness, sigma):
    """Add a blurred, coloured line from ``a`` to ``b`` (additive light) in place."""
    h, w = frame.shape[:2]
    margin = int(thickness + 3 * sigma)
    x0 = max(min(a[0], b[0]) - margin, 0)
    y0 = max(min(a[1], b[1]) - margin, 0)
    x1 = min(max(a[0], b[0]) + margin, w)
    y1 = min(max(a[1], b[1]) + margin, h)
    if x0 >= x1 or y0 >= y1:
        return
    # the glow is soft anyway: draw and blur at half resolution, then scale up
    hw, hh = max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)
    canvas = np.zeros((hh, hw, 3), np.uint8)
    cv2.line(canvas, ((a[0] - x0) // 2, (a[1] - y0) // 2), ((b[0] - x0) // 2, (b[1] - y0) // 2),
             color, max(1, thickness // 2), cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (0, 0), max(0.5, sigma / 2))
    canvas = cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    roi = frame[y0:y1, x0:x1]
    cv2.add(roi, canvas, dst=roi)


def _draw_weapon(frame, saber, color):
    """Material grip and a single controlled emissive halo, no marker-colour wash."""
    if saber is None:
        return frame
    s = config.SCALE
    tip, emitter, pommel = saber
    # When the pen points at the camera its tip can be inside the handle.
    visible=np.dot(np.subtract(tip,emitter),np.subtract(emitter,pommel))>0
    if visible:
        _add_glow(frame, tip, emitter, color, max(4,int(18*s)), max(1.,5*s))
        cv2.line(frame, tip, emitter, color, max(4,int(14*s)),cv2.LINE_AA)
        cv2.line(frame, tip, emitter, (255,255,255), max(3,int(8*s)),cv2.LINE_AA)
    if saber_art.draw(frame,saber,s):
        return frame

    # hilt along the saber axis: dark metal grip with ridges, bright emitter ring
    # at the blade end, lighter pommel cap at the other end
    width = max(6, int(18 * s))
    length = math.dist(emitter, pommel) or 1.0
    d = ((pommel[0] - emitter[0]) / length, (pommel[1] - emitter[1]) / length)
    perp = (-d[1], d[0])

    def across(t, half, color, thickness):
        cx, cy = emitter[0] + d[0] * length * t, emitter[1] + d[1] * length * t
        cv2.line(frame, (int(round(cx - perp[0] * half)), int(round(cy - perp[1] * half))),
                 (int(round(cx + perp[0] * half)), int(round(cy + perp[1] * half))),
                 color, thickness, cv2.LINE_AA)

    cv2.line(frame, emitter, pommel, (60, 60, 65), width)
    for t in (0.35, 0.5, 0.65, 0.8):
        across(t, width / 2, (25, 25, 30), max(1, int(3 * s)))
    across(0.04, width / 2 + max(1, int(3 * s)), (190, 190, 195), max(2, int(6 * s)))
    across(0.96, width / 2, (120, 120, 125), max(2, int(5 * s)))
    return frame


def glove_occlusion(parts, side, sprite, shape, saber):
    """Only the current drawn glove may overlap the grip; no stale hand mask."""
    if saber is None or sprite is None or side is None:return None
    from suit_rig import _warp_roi
    for part in parts or ():
        if part.get('name')!='glove_'+side:continue
        grip=glove_grips([part]).get(side)
        centre=(np.asarray(saber.hilt_top)+saber.hilt_bottom)/2
        if grip is None or math.dist(centre,grip)>12*config.SCALE:return None
        if part.get('matrix') is None:return None
        result=_warp_roi(sprite.image[:,:,3],part['matrix'],shape,sprite.bounds)
        if result is not None:
            x,y,_,_,alpha=result
            return x,y,alpha
    return None


def _grip_patch(frame,saber,occlusion):
    """Save a narrow, rounded finger band from the already rendered real hand.

    This is a conservative visual overlap, not a reconstruction of finger pose.
    The alpha comes from the CURRENT glove/skin; background pixels never cover it.
    """
    if occlusion is None:return None
    ox,oy,mask=occlusion
    grip=(np.asarray(saber.hilt_top,float)+saber.hilt_bottom)/2
    radius=max(3,round(18*config.SCALE))
    x0,y0=max(0,ox,round(grip[0])-radius),max(0,oy,round(grip[1])-radius)
    x1,y1=min(frame.shape[1],ox+mask.shape[1],round(grip[0])+radius+1),min(frame.shape[0],oy+mask.shape[0],round(grip[1])+radius+1)
    if x1<=x0 or y1<=y0:return None
    direction=np.subtract(saber.hilt_bottom,saber.hilt_top).astype(float)
    length=float(np.linalg.norm(direction))
    if length<1:return None
    direction/=length
    yy,xx=np.mgrid[y0:y1,x0:x1]
    dx,dy=xx-grip[0],yy-grip[1]
    along=dx*direction[0]+dy*direction[1]
    across=dx*direction[1]-dy*direction[0]
    support=mask[y0-oy:y1-oy,x0-ox:x1-ox].astype(np.float32)/255
    curve=.8*config.SCALE*(across/(11*config.SCALE))**2
    band=np.zeros_like(along)
    for offset in (-4,0,4):
        band=np.maximum(band,np.clip((2.1*config.SCALE-np.abs(along-offset*config.SCALE-curve))/(.7*config.SCALE),0,1))
    band*=np.clip((12*config.SCALE-np.abs(across))/(2*config.SCALE),0,1)
    alpha=(support*band)[:,:,None]
    raw=frame[y0:y1,x0:x1].copy()
    # Small contact shadow behind the grip, restricted to current hand pixels.
    shadow=support*np.exp(-((across/(12*config.SCALE))**4+(along/(12*config.SCALE))**4))*.10
    frame[y0:y1,x0:x1]=(raw*(1-shadow[:,:,None])).astype(np.uint8)
    return x0,y0,raw,alpha


def draw_saber(frame, saber, color=None, *, occlusion=None, opacity=1.):
    """Draw a grounded weapon, optionally fading through a short visual dropout."""
    if saber is None or opacity<=0:return frame
    color=config.SABER_COLOR if color is None else color
    snapshot=None
    if opacity<1:
        points=np.asarray(saber);margin=max(8,round(40*config.SCALE))
        x0,y0=np.maximum(points.min(axis=0)-margin,0)
        x1,y1=np.minimum(points.max(axis=0)+margin+1,(frame.shape[1],frame.shape[0]))
        if x1>x0 and y1>y0:snapshot=(x0,y0,x1,y1,frame[y0:y1,x0:x1].copy())
    patch=_grip_patch(frame,saber,occlusion)
    _draw_weapon(frame,saber,color)
    if patch is not None:
        x,y,raw,alpha=patch;h,w=raw.shape[:2];roi=frame[y:y+h,x:x+w]
        roi[:]=np.rint(raw*alpha+roi*(1-alpha)).astype(np.uint8)
    if snapshot is not None:
        x0,y0,x1,y1,raw=snapshot;roi=frame[y0:y1,x0:x1]
        roi[:]=np.rint(roi*opacity+raw*(1-opacity)).astype(np.uint8)
    return frame
