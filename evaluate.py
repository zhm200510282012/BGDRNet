"""Optional reviewer evaluation entry point. Not executed during packaging."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,json,sys,zipfile
import numpy as np
from PIL import Image
import torch,cv2
import evaluation_helpers as h
R=Path(__file__).resolve().parent
def main():
 index=json.loads((R/'checkpoint_index.json').read_text())
 p=argparse.ArgumentParser();p.add_argument('--job',choices=list(index),required=True);p.add_argument('--dataset',choices=['GSD_test','GEGD2'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--gsd-root',type=Path);p.add_argument('--gegd2-zip',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 assert not a.out.exists(),'Refuse overwrite'
 info=index[a.job];sys.path.insert(0,str(R/'jobs'/a.job))
 import model,runtime
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2);runtime.seed_all(42)
 net=model.TinyGlass(info['variant'],weights=None,training_heads=False)
 # Only load the author's trusted deployment checkpoint, not arbitrary pickle files.
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 net.load_state_dict(ck['model'],strict=True);assert sum(p.numel() for p in net.parameters())==info['parameters']
 net=net.cuda().eval();del ck
 manifest=list(csv.DictReader((R/'splits'/('gsd_test.csv' if a.dataset=='GSD_test' else 'gegd2.csv')).open(encoding='utf-8-sig')))
 if a.dataset=='GEGD2':assert a.gegd2_zip and a.gegd2_zip.is_file()
 else:assert a.gsd_root and a.gsd_root.is_dir()
 a.out.mkdir(parents=True);rows=[]
 z=zipfile.ZipFile(a.gegd2_zip) if a.dataset=='GEGD2' else None
 try:
  for row in manifest:
   if z:x,truth=h.read_pair(z,row)
   else:
    with Image.open(a.gsd_root/row['image_path']) as im:x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
    with Image.open(a.gsd_root/row['mask_path']) as im:truth=np.asarray(im)>0
   rows.append({'sample_id':row['sample_id'],**h.measure(h.forward(net,x),truth,runtime.boundary_f1)})
 finally:
  if z:z.close()
 h.write_chunk(a.out,rows,0);h.write_json(a.out/'result.json',{'metrics':h.aggregate(rows),'checkpoint':str(a.checkpoint),'threshold':.5,'precision':'CUDA BF16, FP32 weights','input_size':256,'job':a.job,'dataset':a.dataset})
 print('Evaluation completed',a.out)
if __name__=='__main__':main()
