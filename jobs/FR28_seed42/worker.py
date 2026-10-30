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
VARIANT=e.CFG['variant_order'][0]
PARAMS={'FR21':277780,'FR28':277780}

def save(name,obj):e.save_json(R/name,obj)
def training_batch():
    ids=json.loads(Path('D:/SIVP/glass_fr12_regional_context_20261028/logs/source_context_diagnostic.json').read_text(encoding='utf-8'))['fixed_training_ids'][:8]
    with (R/'manifest.csv').open(encoding='utf-8-sig') as f:lookup={r['sample_id']:r for r in csv.DictReader(f)}
    xs=[];ys=[]
    for i in ids:
        r=lookup[i];assert r['split']==r['official_split']=='train'
        with Image.open(r['image_path']) as im:xs.append(np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy())
        with Image.open(r['mask_path']) as im:ys.append((np.asarray(im.resize((256,256),Image.Resampling.NEAREST))>0).copy())
    return torch.from_numpy(np.stack(xs)).permute(0,3,1,2).contiguous().cuda().float()/255,torch.from_numpy(np.stack(ys))[:,None].cuda().float()

def smoke():
    assert e.CFG['epochs']==40 and not e.CFG['task_warmstart']
    e.validate_manifest();x,y=training_batch();e.runtime.seed_all(42)
    model=TinyGlass(VARIANT,e.CFG['weights'],training_heads=True).cuda().eval();rng=torch.get_rng_state().clone()
    e.runtime.seed_all(42);ref=TinyGlass('FR21',e.CFG['weights'],training_heads=True).cuda().eval()
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(model.state_dict()[k],v) for k,v in ref.state_dict().items())
    pretrained=torch.load(e.CFG['weights'],map_location='cpu',weights_only=True)
    assert all(torch.equal(v.cpu(),pretrained['features.'+k]) for k,v in model.encoder.state_dict().items())
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=model(x,with_aux=True);b=ref(x,with_aux=True)
        assert torch.equal(a['logits'],b['logits']) and all(torch.equal(u,v) for u,v in zip(a['edge'],b['edge'])) and all(torch.equal(u,v) for u,v in zip(a['body'],b['body']))
    del a,b,ref,pretrained;gc.collect();torch.cuda.empty_cache()
    opt=torch.optim.AdamW(model.parameters(),lr=e.CFG['lr'],weight_decay=e.CFG['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40)
    steps=[]
    for step in range(4):
        t=time.perf_counter();loss,norm,meta=e.checked_update(model,opt,x,y);torch.cuda.synchronize()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if VARIANT in ['FR21','FR28'] and step>0:assert model.region_low_projection[0].weight.grad.norm()>0
        if VARIANT=='FR28' and step>0:
            grad=model.fusers[0].correction[0].weight.grad
            assert grad[:32].norm()>0 and grad[32:].norm()>0
        steps.append({'seconds':time.perf_counter()-t,'loss':loss,'gradient_norm':norm})
    path=R/'smoke/last.pt';e.save_checkpoint(path,e.make_checkpoint(model,opt,scheduler,0,-1))
    saved=torch.load(path,map_location='cpu',weights_only=False);e.verify_checkpoint(saved,VARIANT)
    assert saved['scheduler']['T_max']==40
    other=TinyGlass(VARIANT,training_heads=True).cuda();other.load_state_dict(saved['model'],strict=True)
    opt2=torch.optim.AdamW(other.parameters(),lr=e.CFG['lr'],weight_decay=e.CFG['weight_decay']);opt2.load_state_dict(saved['optimizer'])
    scheduler2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=40);scheduler2.load_state_dict(saved['scheduler'])
    for net,optim,sched in [(model,opt,scheduler),(other,opt2,scheduler2)]:
        e.runtime.restore_rng(saved['rng']);e.checked_update(net,optim,x,y);sched.step()
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),other.parameters()))
    assert scheduler.state_dict()==scheduler2.state_dict()
    model.eval();deploy=model.deployment_copy().cuda().eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.equal(model(x),deploy(x))
    assert count_parameters(deploy)==PARAMS[VARIANT]
    result={'status':'PASS','identity':e.CFG['run_identity'],'epochs':40,'parameters':PARAMS[VARIANT],'real_training_images':8,
      'initial_common_FR21_tensors_and_RNG_equal':True,'initial_BF16_mask_body_edge_outputs_equal':True,'pretrained_encoder_tensor_verified':True,
      'finite_loss_all_parameter_gradients':True,'resume_next_update_and_scheduler_exact':True,'deployment_exact':True,
      'only_ImageNet_loaded_for_fresh_initialization':True,'old_task_or_smoke_used_to_initialize_formal':'NO',
      'steps':steps,'test_access':False,'plots':False}
    save('logs/smoke.json',result);print(json.dumps(result),flush=True)

def train():
    assert json.loads((R/'logs/smoke.json').read_text())['status']=='PASS'
    save('logs/status.json',{'status':'TRAINING','identity':e.CFG['run_identity'],'epochs':40,'pid':os.getpid()})
    data=e.data_cache();result=e.train_one(VARIANT,data)
    save('logs/status.json',{'status':'COMPLETE','identity':e.CFG['run_identity'],'best_epoch':result['best_epoch']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2);e.runtime.seed_all(42)
    if a.mode=='smoke':smoke()
    else:train()
