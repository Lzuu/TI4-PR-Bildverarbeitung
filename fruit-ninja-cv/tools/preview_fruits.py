"""Reproducible sprite/gallery preview and warm renderer benchmark, no camera."""
from pathlib import Path
import json
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
import fruit_sprites
from fruit import Fruit
import config


def main():
    output=ROOT.parent/'docs/fruit-previews';output.mkdir(parents=True,exist_ok=True)
    yy,xx=np.mgrid[:540,:960]
    scene=np.empty((540,960,3),np.uint8)
    scene[:]=(31,20,15)
    scene[:,:,0]+=(yy/30).astype(np.uint8)
    rng=np.random.default_rng(11)
    for x,y in rng.integers((0,0),(960,540),(90,2)):
        cv2.circle(scene,(int(x),int(y)),1,(130,122,115),-1)
    cv2.putText(scene,'FRUIT NINJA / NEW ARTWORK',(40,45),cv2.FONT_HERSHEY_SIMPLEX,.7,(230,235,240),1,cv2.LINE_AA)
    gallery=[]
    for i,name in enumerate(fruit_sprites.NAMES):
        tile=np.full((260,300,3),(38,30,23),np.uint8)
        fruit_sprites.draw(tile,name,(150,150) if name=='bomb' else (150,130),60 if name=='bomb' else 76)
        fruit_sprites.draw(tile,name,(240,224),26)
        cv2.putText(tile,name.upper(),(18,242),cv2.FONT_HERSHEY_SIMPLEX,.6,(225,230,235),1,cv2.LINE_AA)
        gallery.append(tile)
        fruit_sprites.draw(scene,name,(90+i*155,210),46)
        fruit_sprites.draw(scene,name,(90+i*155,400),35,cut_angle=.45,spread=18)
    cv2.imwrite(str(output/'gallery.png'),np.vstack((np.hstack(gallery[:3]),np.hstack(gallery[3:]))))
    cv2.imwrite(str(output/'game-preview.png'),scene)
    # Representative loaded game objects, mixed whole/sliced; includes frame copy.
    fruits=[]
    for i in range(16):
        f=Fruit(960,540);f.name=fruit_sprites.NAMES[i%6]
        f.is_bomb=f.name=='bomb';f.x=55+(i%8)*120;f.y=160+(i//8)*230
        f.radius=28;f.sliced=i>=8;f.cut_angle=.45;f.slice_time=.15
        fruits.append(f)
    def render():
        frame=scene.copy()
        for f in fruits:f.draw(frame)
    for _ in range(5):render()
    values=[]
    for _ in range(100):
        start=time.perf_counter();render();values.append((time.perf_counter()-start)*1000)
    result=dict(objects=16,frame=[960,540],iterations=100,median_ms=round(statistics.median(values),3),
                p95_ms=round(float(np.percentile(values,95)),3),camera=False,display=False,
                cache=fruit_sprites.get_sprite.cache_info()._asdict())
    (output/'measurements.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
