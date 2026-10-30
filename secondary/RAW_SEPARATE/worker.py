import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,gc,importlib.util,json,time
import numpy as np
from PIL import Image
import torch
import cv2
import experiment as e
from model import TinyGlass,count_parameters
R=Path(__file__).resolve().parent
PHASE=Path('D:/SIVP/glass_fr29_factorial_evidence_20261029')
VARIANT=e.CFG['variant_order'][0]
PARENT={'RAW_SHARED':'FR28','RAW_SEPARATE':'FR29'}[VARIANT]
PARAMS={'RAW_SHARED':277780,'RAW_SEPARATE':278804}

def training_batch():
    rows=list(csv.DictReader((R/'manifest.csv').open(encoding='utf-8')))
    rows=[r for r in rows if r['split']=='train'][:8];xs=[];ys=[]
    for r in rows:
        with Image.open(r['image_path']) as im:xs.append(np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy())
        with Image.open(r['mask_path']) as im:ys.append((np.asarray(im.resize((256,256),Image.Resampling.NEAREST))>0).copy())
    return torch.from_numpy(np.stack(xs)).permute(0,3,1,2).contiguous().cuda().float()/255,torch.from_numpy(np.stack(ys))[:,None].cuda().float()

def reference_equivalence(x):
    f=json.loads((PHASE/'execution.frozen.json').read_text(encoding='utf-8'));evidence={}
    for n,ref in f['references'].items():
        root=Path(ref['path']);ck=torch.load(root/f'runs/{n}/best.pt',map_location='cpu',weights_only=False)
        assert ck['manifest']==f['manifest_text'] and ck['config']==ref['config']
        assert all((root/k).read_text(encoding='utf-8')==v for k,v in ck['source_text'].items())
        for k in ['fr12_frozen.py','fr11_frozen.py','flow_base_frozen.py','fr2_frozen.py','fna_frozen.py','pilot_frozen.py','s2_frozen.py']:
            assert (R/k).read_text(encoding='utf-8')==ref['sources'][k]
        spec=importlib.util.spec_from_file_location('original_'+n,root/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        a=mod.TinyGlass(n,training_heads=True).cuda().eval();a.load_state_dict(ck['model'],strict=True)
        b=TinyGlass(n,training_heads=True).cuda().eval();b.load_state_dict(ck['model'],strict=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):diff=float((a(x)-b(x)).abs().max())
        assert diff==0
        evidence[n]={'checkpoint_epoch':ck['epoch'],'canonical_model_output_max_abs_diff':diff,'source_manifest_config_valid':True}
        del a,b,ck;gc.collect();torch.cuda.empty_cache()
    return evidence

def smoke():
    e.validate_manifest();x,y=training_batch();assert x.shape==(8,3,256,256) and y.shape==(8,1,256,256)
    assert set(torch.unique(y).tolist())=={0.,1.}
    e.runtime.seed_all(42);net=TinyGlass(VARIANT,e.CFG['weights'],training_heads=True).cuda().eval();rng=torch.get_rng_state().clone()
    e.runtime.seed_all(42);ref=TinyGlass(PARENT,e.CFG['weights'],training_heads=True).cuda().eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert net.state_dict().keys()==ref.state_dict().keys() and all(torch.equal(net.state_dict()[k],v) for k,v in ref.state_dict().items())
    w=torch.load(e.CFG['weights'],map_location='cpu',weights_only=True)
    assert all(torch.equal(v.cpu(),w['features.'+k]) for k,v in net.encoder.state_dict().items())
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=net(x,capture=True);b=ref(x,capture=True)
        assert a['logits'].shape==y.shape
        assert torch.equal(a['trace']['difference0'],a['trace']['feature2'])
        assert torch.equal(b['trace']['difference0'],b['trace']['feature2']-b['trace']['reference_high0'])
        for k in ['feature0','feature1','feature2','feature3','adapted','region0']:
            assert torch.equal(a['trace'][k],b['trace'][k]),k
    del a,b,ref,w
    net.train();net.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        seg=torch.nn.functional.binary_cross_entropy_with_logits(net(x,with_aux=True)['logits'],y)
    seg.backward();assert all(p.grad is None for head in net.edge_heads for p in head.parameters())
    net.zero_grad(set_to_none=True)
    opt=torch.optim.AdamW(net.parameters(),lr=e.CFG['lr'],weight_decay=e.CFG['weight_decay']);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40)
    steps=[]
    for i in range(4):
        torch.cuda.synchronize();t=time.perf_counter();v,n,meta=e.checked_update(net,opt,x,y);torch.cuda.synchronize()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        if i>0:
            assert net.fusers[0].correction[0].weight.grad.norm()>0
            assert net.region_low_projection[0].weight.grad.norm()>0
        steps.append({'loss':v,'gradient_norm':n,'seconds':time.perf_counter()-t,**meta})
    p=R/'smoke/last.pt';e.save_checkpoint(p,e.make_checkpoint(net,opt,sch,0,-1))
    ck=torch.load(p,map_location='cpu',weights_only=False);e.verify_checkpoint(ck,VARIANT)
    other=TinyGlass(VARIANT,training_heads=True).cuda();other.load_state_dict(ck['model'])
    opt2=torch.optim.AdamW(other.parameters(),lr=e.CFG['lr'],weight_decay=e.CFG['weight_decay']);opt2.load_state_dict(ck['optimizer'])
    sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=40);sch2.load_state_dict(ck['scheduler'])
    for m,o,s in [(net,opt,sch),(other,opt2,sch2)]:e.runtime.restore_rng(ck['rng']);e.checked_update(m,o,x,y);s.step()
    assert all(torch.equal(a,b) for a,b in zip(net.state_dict().values(),other.state_dict().values()))
    assert sch.state_dict()==sch2.state_dict()
    net.eval();deploy=net.deployment_copy().cuda().eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.equal(net(x),deploy(x))
    assert count_parameters(deploy)==PARAMS[VARIANT]
    result={'status':'PASS','variant':VARIANT,'inference_parameters':count_parameters(deploy),'training_parameters':count_parameters(net),
        'real_training_images':8,'initial_parent_tensors_and_RNG_equal':True,'pretrained_encoder_tensors_exact':True,
        'first_stage_raw_detail_dataflow_verified':True,'coarse_features_and_region_before_fusion_equal_parent':True,
        'finite_loss_all_parameter_gradients':True,'segmentation_edge_gradient_blocked':True,
        'resume_next_update_and_scheduler_exact':True,'deployment_output_exact':True,'steps':steps,
        'nominal_training_step_seconds':float(np.median([s['seconds'] for s in steps[1:]])),
        'test_access':False,'formal_warmstart_from_smoke':False}
    del net,other,opt,opt2,sch,sch2,deploy,ck;gc.collect();torch.cuda.empty_cache()
    result['reference_equivalence']=reference_equivalence(x)
    e.save_json(R/'logs/smoke.json',result);print(json.dumps(result),flush=True)

def train():
    assert json.loads((R/'logs/smoke.json').read_text(encoding='utf-8'))['status']=='PASS'
    d=e.data_cache();result=e.train_one(VARIANT,d)
    h=[json.loads(x) for x in (R/f'runs/{VARIANT}/epochs.jsonl').read_text(encoding='utf-8').splitlines() if x]
    assert [x['epoch'] for x in h]==list(range(1,41)) and all(x['global_step']==271*x['epoch'] for x in h)
    best=max(h,key=lambda x:(x['validation']['mIoU'],-x['epoch']))
    assert best['epoch']==result['best_epoch'] and best['validation']==result['validation']
    ck=torch.load(R/f'runs/{VARIANT}/last.pt',map_location='cpu',weights_only=False);e.verify_checkpoint(ck,VARIANT)
    assert ck['epoch']==40 and ck['scheduler']['last_epoch']==40
    fields=['epoch','mean_training_loss','lr','mIoU','glass_IoU','MAE','boundary_F1','non_glass_FPR','glass_FNR']
    with (R/f'runs/{VARIANT}/curves.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in h:w.writerow({'epoch':row['epoch'],'mean_training_loss':row['mean_training_loss'],'lr':row['lr_used'][0],**{k:row['validation'][k] for k in fields[3:]}})
    e.save_json(R/'logs/completed_audit.json',{'status':'PASS','epochs':40,'updates':10840,'best_epoch':best['epoch'],'resume_checkpoint_valid':True,'test_access':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2);e.runtime.seed_all(42)
    {'smoke':smoke,'train':train}[a.mode]()
