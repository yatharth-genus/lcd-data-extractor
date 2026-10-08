from __future__ import annotations
import argparse, base64, json, os, sys, traceback
from pathlib import Path
import cv2
import numpy as np

def emit(x):
    sys.stdout.write(json.dumps(x, ensure_ascii=False) + "\n"); sys.stdout.flush()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--det-model",type=Path,required=True); p.add_argument("--rec-model",type=Path,required=True); p.add_argument("--dictionary",type=Path,required=True); a=p.parse_args()
    root=Path(os.environ["PADDLEOCR_ROOT"]); sys.path.insert(0,str(root))
    from tools.infer.predict_system import TextSystem
    from tools.infer.utility import parse_args
    old=sys.argv[:]
    try:
        sys.argv=["predict_system.py","--det_model_dir",str(a.det_model),"--rec_model_dir",str(a.rec_model),"--rec_char_dict_path",str(a.dictionary),"--use_gpu","false","--use_angle_cls","false","--use_space_char","true","--rec_image_shape","3,48,320","--det_algorithm","DB","--rec_algorithm","SVTR_LCNet","--det_limit_type","max","--det_limit_side_len","1280","--det_db_thresh","0.20","--det_db_box_thresh","0.45","--det_db_unclip_ratio","1.40","--drop_score","0.0","--show_log","false"]
        args=parse_args()
    finally: sys.argv=old
    system=TextSystem(args); emit({"status":"ready"})
    for line in sys.stdin:
        try:
            req=json.loads(line)
            if req.get("command")=="shutdown": emit({"status":"stopped"}); return
            image=cv2.imdecode(np.frombuffer(base64.b64decode(req["image_base64"]),dtype=np.uint8),cv2.IMREAD_COLOR)
            boxes,recs,_=system(image); out=[]
            if boxes is not None and recs is not None:
                for box,rec in zip(boxes,recs):
                    text,score=rec; out.append({"polygon":[[int(round(float(x))),int(round(float(y)))] for x,y in box],"text":str(text),"confidence":float(score)})
            emit({"status":"success","results":out})
        except Exception as e: emit({"status":"error","error":str(e),"traceback":traceback.format_exc()})
if __name__=="__main__": main()
