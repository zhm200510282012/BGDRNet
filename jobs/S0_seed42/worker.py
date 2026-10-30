import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,gc,importlib.util,json,time
import numpy as np,torch,cv2
from PIL import Image
import experiment as e
from model import TinyGlass,count_parameters
R=Path(__file__).resolve().parent
def batch():
    with (R/'manifest.csv').open(encoding='utf-8-sig') as f:rows=[r for r in csv.DictReader(f) if r['split']=='train'][:8]
    xs=[];ys=[]
    for r in rows:
        assert r['official_split']=='train'
        with Image.open(r['image_path']) as im:xs.append(np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy())
        with Image.open(r['mask_path']) as im:ys.append((np.asarray(im.resize((256,256),Image.Resampling.NEAREST))>0).copy())
    return torch.from_numpy(np.stack(xs)).permute(0,3,1,2).contiguous().cuda().float()/255,torch.from_numpy(np.stack(ys))[:,None].cuda().float()
def smoke():
    assert e.CFG['epochs']==40 and e.CFG['seed']==42 and not e.CFG['task_warmstart']
    e.validate_manifest();x,y=batch();e.runtime.seed_all(42)
    net=TinyGlass('S0',e.CFG['weights']).cuda().eval()
    original=Path('D:/SIVP/glass_light_mvp_20261025/model.py')
    assert (R/'model.py').read_bytes()==original.read_bytes()
    spec=importlib.util.spec_from_file_location('original_s0_for_smoke',original);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    e.runtime.seed_all(42);ref=m.TinyGlass('S0',e.CFG['weights']).cuda().eval()
    assert all(torch.equal(v,ref.state_dict()[k]) for k,v in net.state_dict().items())
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=net(x);b=ref(x);assert a.shape==(8,1,256,256) and torch.equal(a,b)
    del ref,a,b
    weights=torch.load(e.CFG['weights'],map_location='cpu',weights_only=True)
    assert all(torch.equal(v.cpu(),weights['features.'+k]) for k,v in net.encoder.state_dict().items())
    del weights
    assert len(net.edge_heads)==len(net.body_heads)==0
    opt=torch.optim.AdamW(net.parameters(),lr=e.CFG['lr'],weight_decay=e.CFG['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40)
    steps=[]
    for i in range(4):
        t=time.perf_counter();loss,norm,meta=e.checked_update(net,opt,x,y);torch.cuda.synchronize()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        assert meta['edge_loss']==meta['region_loss']==0
        steps.append(dict(seconds=time.perf_counter()-t,loss=loss,gradient_norm=norm))
    e.save_checkpoint(R/'smoke/last.pt',e.make_checkpoint(net,opt,sched,0,-1))
    ck=torch.load(R/'smoke/last.pt',map_location='cpu',weights_only=False);e.verify_checkpoint(ck,'S0')
    other=TinyGlass('S0').cuda();other.load_state_dict(ck['model'],strict=True)
    opt2=torch.optim.AdamW(other.parameters(),lr=e.CFG['lr'],weight_decay=e.CFG['weight_decay']);opt2.load_state_dict(ck['optimizer'])
    sched2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=40);sched2.load_state_dict(ck['scheduler'])
    for n,o,s in [(net,opt,sched),(other,opt2,sched2)]:
        e.runtime.restore_rng(ck['rng']);e.checked_update(n,o,x,y);s.step()
    assert all(torch.equal(a,b) for a,b in zip(net.parameters(),other.parameters()))
    assert sched.state_dict()==sched2.state_dict()
    net.eval();deployed=net.deployment_copy().cuda().eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):assert torch.equal(net(x),deployed(x))
    assert count_parameters(net)==count_parameters(deployed)==255969
    result=dict(status='PASS',identity=e.CFG['run_identity'],real_training_images=8,parameters=255969,
        original_S0_source_and_initial_tensors_and_outputs_exact=True,pretrained_encoder_tensor_verified=True,
        output_shape=[8,1,256,256],finite_loss_all_parameter_gradients=True,auxiliary_heads_and_losses_absent=True,
        resume_next_update_and_scheduler_exact=True,deployment_exact=True,formal_uses_smoke_weights=False,
        formal_uses_old_task_weights=False,steps=steps,config=e.CFG,source_text=e.SOURCES,test_access=False,plots=False)
    e.save_json(R/'logs/smoke.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_text','config']}),flush=True)
def train():
    smoke= json.loads((R/'logs/smoke.json').read_text(encoding='utf-8'))
    assert smoke['status']=='PASS' and smoke['config']==e.CFG and smoke['source_text']==e.SOURCES
    data=e.data_cache();result=e.train_one('S0',data)
    p=R/'runs/S0';history=[json.loads(x) for x in (p/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if x]
    assert [x['epoch'] for x in history]==list(range(1,41))
    assert all(x['global_step']==x['epoch']*271 and x['samples_seen_this_epoch']==2168 for x in history)
    best=max(history,key=lambda x:x['validation']['mIoU']);assert best['epoch']==result['best_epoch']
    ck=torch.load(p/'last.pt',map_location='cpu',weights_only=False);e.verify_checkpoint(ck,'S0')
    assert ck['epoch']==40 and ck['scheduler']['T_max']==40 and ck['scheduler']['last_epoch']==40
    assert all(int(v['step'])==10840 for v in ck['optimizer']['state'].values())
    with (p/'curves.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['epoch','mean_training_loss','lr','mIoU','glass_IoU','boundary_F1','non_glass_FPR','glass_FNR','MAE','seconds']);w.writeheader()
        for h in history:w.writerow(dict(epoch=h['epoch'],mean_training_loss=h['mean_training_loss'],lr=h['lr_used'][0],seconds=h['seconds'],**{k:h['validation'][k] for k in ['mIoU','glass_IoU','boundary_F1','non_glass_FPR','glass_FNR','MAE']}))
    e.save_json(R/'logs/completed_audit.json',dict(status='PASS',epochs=40,updates=10840,best_epoch=best['epoch'],source_config_manifest_match=True,resume_state_present=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','train'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2);e.runtime.seed_all(42)
    if a.mode=='smoke':smoke()
    else:train()
