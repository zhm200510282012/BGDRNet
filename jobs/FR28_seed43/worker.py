import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,gc,json,time
import numpy as np
from PIL import Image
import torch
import cv2
import experiment as e
from model import TinyGlass,count_parameters
R=Path(__file__).resolve().parent
VARIANT=e.CFG['variant_order'][0];SEED=e.CFG['seed']
def batch():
    rows=[r for r in csv.DictReader((R/'manifest.csv').open(encoding='utf-8')) if r['split']=='train'][:8];xs=[];ys=[]
    for r in rows:
        with Image.open(r['image_path']) as im:xs.append(np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy())
        with Image.open(r['mask_path']) as im:ys.append((np.asarray(im.resize((256,256),Image.Resampling.NEAREST))>0).copy())
    return torch.from_numpy(np.stack(xs)).permute(0,3,1,2).contiguous().cuda().float()/255,torch.from_numpy(np.stack(ys))[:,None].cuda().float()
def smoke():
    assert SEED in [43,44] and e.CFG['epochs']==40 and not e.CFG['task_warmstart']
    e.validate_manifest();x,y=batch();e.runtime.seed_all(SEED)
    m=TinyGlass(VARIANT,e.CFG['weights'],training_heads=True).cuda().eval();rng=torch.get_rng_state().clone()
    e.runtime.seed_all(SEED);ref=TinyGlass('FR29' if VARIANT=='FR28' else 'FR28',e.CFG['weights'],training_heads=True).cuda().eval()
    assert torch.equal(rng,torch.get_rng_state())
    a,b=m.state_dict(),ref.state_dict();common=a.keys()&b.keys()
    assert all(torch.equal(a[k],b[k]) for k in common)
    assert (a.keys()^b.keys())=={'fusers.0.correction.value_projection.weight'}
    w=torch.load(e.CFG['weights'],map_location='cpu',weights_only=True)
    assert all(torch.equal(v.cpu(),w['features.'+k]) for k,v in m.encoder.state_dict().items())
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=m(x,with_aux=True);other=ref(x,with_aux=True)
        assert out['logits'].shape==(8,1,256,256) and torch.equal(out['logits'],other['logits'])
        assert all(torch.equal(u,v) for key in ['body','edge'] for u,v in zip(out[key],other[key]))
    del a,b,w,out,other,ref
    m.train();m.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):z=torch.nn.functional.binary_cross_entropy_with_logits(m(x,with_aux=True)['logits'],y)
    z.backward();assert all(p.grad is None for head in m.edge_heads for p in head.parameters());m.zero_grad(set_to_none=True)
    opt=torch.optim.AdamW(m.parameters(),lr=e.CFG['lr'],weight_decay=e.CFG['weight_decay']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40)
    steps=[]
    for i in range(4):
        torch.cuda.synchronize();t=time.perf_counter();v,n,meta=e.checked_update(m,opt,x,y);torch.cuda.synchronize()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        steps.append({'seconds':time.perf_counter()-t,'loss':v,'gradient_norm':n})
    e.save_checkpoint(R/'smoke/last.pt',e.make_checkpoint(m,opt,sch,0,-1));ck=torch.load(R/'smoke/last.pt',map_location='cpu',weights_only=False);e.verify_checkpoint(ck,VARIANT)
    n=TinyGlass(VARIANT,training_heads=True).cuda();n.load_state_dict(ck['model'])
    opt2=torch.optim.AdamW(n.parameters(),lr=e.CFG['lr'],weight_decay=e.CFG['weight_decay']);opt2.load_state_dict(ck['optimizer'])
    sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=40);sch2.load_state_dict(ck['scheduler'])
    for net,o,s in [(m,opt,sch),(n,opt2,sch2)]:e.runtime.restore_rng(ck['rng']);e.checked_update(net,o,x,y);s.step()
    assert all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),n.state_dict().values())) and sch.state_dict()==sch2.state_dict()
    deploy=m.deployment_copy().cuda().eval();m.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.equal(m(x),deploy(x))
    expected=277780 if VARIANT=='FR28' else 278804;assert count_parameters(deploy)==expected
    result={'status':'PASS','variant':VARIANT,'seed':SEED,'inference_parameters':expected,'real_images':8,
        'same_seed_common_tensors_RNG_and_initial_outputs_exact':True,'pretrained_tensors_exact':True,'all_parameter_gradients_finite':True,
        'segmentation_edge_gradient_blocked':True,'resume_next_update_and_scheduler_exact':True,'deployment_exact':True,
        'steps':steps,'formal_uses_smoke_or_task_weights':False,'test_access':False}
    e.save_json(R/'logs/smoke.json',result);print(json.dumps(result),flush=True)
def train():
    assert json.loads((R/'logs/smoke.json').read_text(encoding='utf-8'))['status']=='PASS'
    result=e.train_one(VARIANT,e.data_cache())
    h=[json.loads(x) for x in (R/f'runs/{VARIANT}/epochs.jsonl').read_text(encoding='utf-8').splitlines() if x]
    assert [x['epoch'] for x in h]==list(range(1,41)) and all(x['global_step']==271*x['epoch'] for x in h)
    best=max(h,key=lambda x:(x['validation']['mIoU'],-x['epoch']));assert best['epoch']==result['best_epoch'] and best['validation']==result['validation']
    ck=torch.load(R/f'runs/{VARIANT}/last.pt',map_location='cpu',weights_only=False);e.verify_checkpoint(ck,VARIANT)
    assert ck['epoch']==40 and ck['scheduler']['last_epoch']==40
    fields=['epoch','mean_training_loss','mIoU','glass_IoU','MAE','boundary_F1','non_glass_FPR','glass_FNR']
    with (R/f'runs/{VARIANT}/curves.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in h:w.writerow({'epoch':row['epoch'],'mean_training_loss':row['mean_training_loss'],**{k:row['validation'][k] for k in fields[2:]}})
    e.save_json(R/'logs/completed_audit.json',{'status':'PASS','variant':VARIANT,'seed':SEED,'epochs':40,'updates':10840,'best_epoch':best['epoch']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2);e.runtime.seed_all(SEED)
    {'smoke':smoke,'train':train}[a.mode]()
