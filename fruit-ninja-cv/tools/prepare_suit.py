"""Download/verify the local pose model and validate packaged EVA assets."""
from pathlib import Path
import argparse
import sys
import socket

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pose_tracking import ensure_model
from suit_rig import load_assets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true',help='Verify existing files without downloads')
    parser.add_argument('--all-models',action='store_true',help='Also prepare the existing face/YOLO models')
    args=parser.parse_args()
    socket.setdefaulttimeout(20)
    model=ensure_model(download=not args.offline)
    detector=ensure_model(download=not args.offline,kind='detector')
    assets,_=load_assets()
    print('Pose model verified:',model)
    print('Pose initializer verified:',detector)
    print('Local RGBA assets:',', '.join(assets))
    print('Decoded sprite pixel memory: %.2f MiB' % (sum(s.image.nbytes+s.premultiplied.nbytes for s in assets.values())/2**20))
    if args.all_models:
        import face_id
        import yolo_seg
        face_id.ensure_models(download=not args.offline)
        yolo_seg.ensure_model(download=not args.offline)
        print('Existing YuNet/SFace/YOLO models verified.')


if __name__=='__main__':
    main()
