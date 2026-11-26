"""Reproduce the packaged comic fruit / sword previews without webcam or GUI."""
from pathlib import Path
import sys,json,time,statistics
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import cv2
import numpy as np
import lightsaber
from preview_fruits import main as preview_fruits


def main():
    preview_fruits()
    dest=ROOT.parent/'docs/comic-art-84';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    poses=[('Vertical',lightsaber.Saber((150,45),(150,300),(150,390))),
           ('Horizontal',lightsaber.Saber((60,200),(235,200),(300,200))),
           ('Diagonal',lightsaber.Saber((310,70),(120,270),(60,330)))]
    for title,pose in poses:
        tile=np.full((440,360,3),(38,28,20),np.uint8)
        lightsaber.draw_saber(tile,pose,(240,210,50))
        cv2.putText(tile,title,(20,420),cv2.FONT_HERSHEY_SIMPLEX,.6,(225,230,235),1,cv2.LINE_AA)
        rows.append(tile)
    cv2.imwrite(str(dest/'laser-sword-poses.png'),np.hstack(rows))
    game=cv2.imread(str(ROOT.parent/'docs/fruit-previews/game-preview.png'))
    lightsaber.draw_saber(game,lightsaber.Saber((700,275),(490,410),(435,445)),(240,210,50))
    cv2.imwrite(str(dest/'game-preview.png'),game)
    frame=np.zeros((540,960,3),np.uint8);pose=poses[2][1]
    for _ in range(5):lightsaber.draw_saber(frame.copy(),pose,(240,210,50))
    timings=[]
    for _ in range(80):
        image=frame.copy();start=time.perf_counter()
        lightsaber.draw_saber(image,pose,(240,210,50))
        timings.append((time.perf_counter()-start)*1000)
    measurements=dict(stage='draw_saber',iterations=80,frame=[960,540],
        median_ms=round(statistics.median(timings),3),p95_ms=round(float(np.percentile(timings,95)),3),
        camera=False,display=False)
    (dest/'measurements.json').write_text(json.dumps(measurements,indent=2),encoding='utf-8')
    print(json.dumps(measurements))

if __name__=='__main__':main()
