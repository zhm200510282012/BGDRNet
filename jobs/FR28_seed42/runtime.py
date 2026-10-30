# Frozen utilities reused from the previous completed screen; main/training code excluded.
import csv,json,math,random,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
BASE=Path(__file__).resolve().parent
DEVICE=torch.device('cuda')


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

class DataCache:
    def __init__(self, size):
        self.size=size
        with (BASE/'manifest.csv').open(encoding='utf-8') as f:
            self.rows=list(csv.DictReader(f))
        self.cached={}
        start=time.monotonic()
        # Cache resized train/val uint8 tensors only; avoid repeated JPEG decoding and SSD writes.
        for i,row in enumerate(self.rows):
            if row['split']=='test':
                continue
            with Image.open(row['image_path']) as im:
                x=np.asarray(im.convert('RGB').resize((size,size),Image.Resampling.BILINEAR)).copy()
            with Image.open(row['mask_path']) as im:
                y=(np.asarray(im.resize((size,size),Image.Resampling.NEAREST))>0).astype(np.uint8)
            self.cached[i]=(torch.from_numpy(x).permute(2,0,1).contiguous(),torch.from_numpy(y[None]).contiguous())
        self.splits={s:[i for i,r in enumerate(self.rows) if r['split']==s] for s in ['train','val','test']}
        print(json.dumps({'event':'cache_ready','seconds':time.monotonic()-start,'counts':{k:len(v) for k,v in self.splits.items()}},ensure_ascii=False),flush=True)

    def batch(self, ids, flips=None):
        pairs=[]
        for i in ids:
            if i in self.cached:
                pairs.append(self.cached[i])
            else:
                row=self.rows[i]
                with Image.open(row['image_path']) as im:
                    x=np.asarray(im.convert('RGB').resize((self.size,self.size),Image.Resampling.BILINEAR)).copy()
                with Image.open(row['mask_path']) as im:
                    y=(np.asarray(im.resize((self.size,self.size),Image.Resampling.NEAREST))>0).astype(np.uint8)
                pairs.append((torch.from_numpy(x).permute(2,0,1).contiguous(),torch.from_numpy(y[None])))
        x=torch.stack([v[0] for v in pairs]).to(DEVICE,dtype=torch.float32).div_(255.)
        y=torch.stack([v[1] for v in pairs]).to(DEVICE,dtype=torch.float32)
        if flips is not None:
            idx=torch.as_tensor(flips,device=DEVICE,dtype=torch.bool)
            x[idx]=x[idx].flip(-1); y[idx]=y[idx].flip(-1)
        return x,y

def boundary_f1(pred,truth):
    kernel=np.ones((3,3),np.uint8)
    pb=pred & ~cv2.erode(pred.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=1).astype(bool)
    tb=truth & ~cv2.erode(truth.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=1).astype(bool)
    npb,ntb=int(pb.sum()),int(tb.sum())
    if npb==0 or ntb==0:
        return float(npb==ntb)
    tol=max(1,math.ceil(.0075*math.hypot(*truth.shape)))
    dt=cv2.distanceTransform((~tb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    dp=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    p=float((dt[pb]<=tol).mean()); r=float((dp[tb]<=tol).mean())
    return 2*p*r/(p+r) if p+r else 0.

@torch.inference_mode()
def evaluate(model,data,split,batch_size):
    model.eval()
    tp=fp=tn=fn=pixels=0; abs_error=0.; bfs=[]
    for start in range(0,len(data.splits[split]),batch_size):
        ids=data.splits[split][start:start+batch_size]
        x,_=data.batch(ids)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(x)
        logits=output[:,0].float().cpu().numpy()
        for j,i in enumerate(ids):
            row=data.rows[i]
            with Image.open(row['mask_path']) as im:
                truth=np.asarray(im)>0
            logit=cv2.resize(logits[j],(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)
            prob=1/(1+np.exp(-np.clip(logit,-80,80)))
            pred=prob>=.5
            tp+=int((pred&truth).sum()); fp+=int((pred&~truth).sum())
            tn+=int((~pred&~truth).sum()); fn+=int((~pred&truth).sum())
            pixels+=truth.size; abs_error+=float(np.abs(prob-truth).sum(dtype=np.float64))
            bfs.append(boundary_f1(pred,truth))
    gi=tp/(tp+fp+fn) if tp+fp+fn else 1.
    bi=tn/(tn+fp+fn) if tn+fp+fn else 1.
    return {'mIoU':(gi+bi)/2,'glass_IoU':gi,'MAE':abs_error/pixels,'boundary_F1':float(np.mean(bfs)),
            'non_glass_FPR':fp/(fp+tn) if fp+tn else None,'images':len(bfs),'TP':tp,'FP':fp,'TN':tn,'FN':fn}

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['cpu']); torch.cuda.set_rng_state_all(r['cuda'])
