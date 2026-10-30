"""One-shot smoke/train/profile jobs. No GPU polling or test-set access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, csv, gc, importlib.util, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models import build, parameters, loss
import runtime

R=Path(__file__).resolve().parent
CFG=json.loads((R/'config.frozen.json').read_text(encoding='utf-8'))
FROZEN=json.loads((R/'execution.frozen.json').read_text(encoding='utf-8'))
CACHE=Path(CFG['cache'])
torch.set_num_threads(CFG['threads'])

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)

def check_sources():
    assert CFG==FROZEN['config']
    assert (R/'manifest.csv').read_text(encoding='utf-8')==FROZEN['manifest']
    assert all((R/k).read_text(encoding='utf-8')==v for k,v in FROZEN['sources'].items()), 'Frozen source changed'
    assert (R/'PROTOCOL.md').read_text(encoding='utf-8')==FROZEN['protocol']

def model(name, pretrained=False):
    if name=='FR29':
        sys.path.insert(0,str(R/'reference'))
        spec=importlib.util.spec_from_file_location('fr29_reference_model',R/'reference/model.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod.TinyGlass('FR29',training_heads=False)
    return build(name,CFG['weights'] if pretrained and name=='LRASPP_prefix' else None)

def ckpath(name):
    return Path(CFG['reference_path'])/'runs/FR29/best.pt' if name=='FR29' else CACHE/name/'best.pt'

def load_best(name):
    ck=torch.load(ckpath(name),map_location='cpu',weights_only=False)
    net=model(name)
    state=ck['model']
    if name=='FR29':
        ref=Path(CFG['reference_path'])
        assert ck['manifest']==FROZEN['manifest']
        assert ck['config']==json.loads((ref/'config.frozen.json').read_text(encoding='utf-8'))
        assert all((ref/k).read_text(encoding='utf-8')==v for k,v in ck['source_text'].items())
        assert all((R/'reference'/k).read_text(encoding='utf-8')==ck['source_text'][k] for k in ['model.py','fr12_frozen.py','fr11_frozen.py','flow_base_frozen.py','fr2_frozen.py','fna_frozen.py','pilot_frozen.py','s2_frozen.py'])
        expected=net.state_dict(); removed=[k for k in state if k not in expected]
        assert all(k.startswith('body_heads.') for k in removed), removed
        state={k:state[k] for k in expected}
        assert ck['epoch']==28 and ck['validation']==json.loads((R/'reference_reuse.json').read_text(encoding='utf-8'))['result']['validation']
    else:verify(ck,name)
    net.load_state_dict(state,strict=True)
    return net.cuda().eval()

def data():
    rows=list(csv.DictReader((R/'manifest.csv').open(encoding='utf-8')))
    assert len(rows)==2710 and len({r['sample_id'] for r in rows})==2710
    assert {r['official_split'] for r in rows}=={'train'}
    assert sum(r['split']=='train' for r in rows)==2168 and sum(r['split']=='val' for r in rows)==542
    for row in rows:
        assert row['split'] in ['train','val']
        for key in ['image_path','mask_path']:
            p=Path(row[key]);assert p.is_file() and p.stat().st_size>0
            assert 'test' not in [x.lower() for x in p.parts]
    result=runtime.DataCache(CFG['size']);assert not result.splits.pop('test')
    return result

def evaluate(net,d):
    result=runtime.evaluate(net,d,'val',CFG['batch_size'])
    result['glass_FNR']=result['FN']/(result['TP']+result['FN'])
    return result

def update(net,opt,x,y):
    net.train();opt.zero_grad(set_to_none=True)
    total,parts=loss(net,x,y);total.backward()
    norm=nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
    opt.step()
    return float(total.detach()),float(norm),parts

def ck_save(p,state):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');torch.save(state,t);t.replace(p)

def checkpoint(net,opt,sch,epoch,best,name):
    return {'model':net.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),
        'epoch':epoch,'best_mIoU':best,'variant':name,'config':CFG,'manifest':FROZEN['manifest'],
        'source_text':FROZEN['sources'],'rng':runtime.rng_state()}

def verify(ck,name):
    assert ck['variant']==name and ck['config']==CFG and ck['manifest']==FROZEN['manifest']
    assert ck['source_text']==FROZEN['sources']

def smoke():
    d=data();x,y=d.batch(d.splits['train'][:8])
    assert x.shape==(8,3,256,256) and y.shape==(8,1,256,256)
    assert set(torch.unique(y).tolist())=={0.,1.} and 0<float(y.mean())<1
    report={'status':'RUNNING','data':'First eight real GSD training pairs, no test access','arms':{}}
    for name in CFG['runnable']:
        runtime.seed_all(CFG['seed']);net=model(name,True).cuda()
        pretrain='NOT_APPLICABLE_SCRATCH'
        if name=='LRASPP_prefix':
            w=torch.load(CFG['weights'],map_location='cpu',weights_only=True)
            assert all(torch.equal(v.cpu(),w['features.'+k]) for k,v in net.encoder.state_dict().items())
            del w;pretrain='ALL_ENCODER_TENSORS_EXACT'
        net.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=net(x)
        assert z.shape==y.shape and torch.isfinite(z).all()
        out_shape=list(z.shape);del z
        opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=40)
        steps=[]
        for i in range(5):
            torch.cuda.synchronize();t=time.perf_counter();v,n,parts=update(net,opt,x,y);torch.cuda.synchronize()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            steps.append({'loss':v,'gradient_norm':n,'seconds':time.perf_counter()-t,**parts})
        out=CACHE/'smoke'/name;state=checkpoint(net,opt,sch,0,-1.,name);ck_save(out/'last.pt',state)
        restored=torch.load(out/'last.pt',map_location='cpu',weights_only=False);verify(restored,name)
        net2=model(name).cuda();net2.load_state_dict(restored['model'])
        opt2=torch.optim.AdamW(net2.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);opt2.load_state_dict(restored['optimizer'])
        sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=40);sch2.load_state_dict(restored['scheduler'])
        assert sch2.state_dict()==sch.state_dict()
        for m,o in [(net,opt),(net2,opt2)]:runtime.restore_rng(restored['rng']);update(m,o,x,y)
        diffs=[float((a-b).abs().max()) for a,b in zip(net.state_dict().values(),net2.state_dict().values()) if a.is_floating_point()]
        maxdiff=max(diffs);assert maxdiff<=1e-6,maxdiff
        assert opt.state_dict()['param_groups']==opt2.state_dict()['param_groups']
        for key,val in opt.state_dict()['state'].items():
            for k,v in val.items():
                v2=opt2.state_dict()['state'][key][k]
                assert torch.equal(v,v2) if isinstance(v,torch.Tensor) else v==v2
        net.eval()
        # Verify deployment serialization preserves the full two-class inference model.
        ck_save(out/'deployment.pt',{'model':net.state_dict(),'smoke_only':True})
        dep=model(name).cuda().eval();dep.load_state_dict(torch.load(out/'deployment.pt',map_location='cpu',weights_only=True)['model'])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):dd=float((net(x)-dep(x)).abs().max())
        assert dd==0
        old=d.splits['val'];d.splits['val']=old[:16]
        t=time.perf_counter();mini=evaluate(net,d);val_seconds=time.perf_counter()-t;d.splits['val']=old
        assert all(np.isfinite(mini[k]) for k in ['mIoU','glass_IoU','MAE','boundary_F1','non_glass_FPR','glass_FNR'])
        estimate=40*(271*np.median([s['seconds'] for s in steps[2:]])+val_seconds*542/16)
        report['arms'][name]={'status':'PASS','training_parameters':parameters(net),'inference_parameters':parameters(dep),
            'output_shape':out_shape,'finite_loss_gradients':True,'all_parameters_have_gradients':True,
            'checkpoint_next_update_max_abs_diff':maxdiff,'optimizer_resume_exact':True,'scheduler_resume_exact':True,
            'deployment_output_max_abs_diff':dd,'pretrained_tensor_transfer':pretrain,'steps':steps,
            'validation_engine_finite_16_images':True,'validation_probe_seconds':val_seconds,
            'formal_seconds_estimate':float(estimate),'estimate_scope':'Approximate steady 8-image update and 16-image native metric cost extrapolation; not a runtime guarantee'}
        print(json.dumps({'event':'smoke_pass','model':name,**report['arms'][name]}),flush=True)
        del net,net2,opt,opt2,sch,sch2,dep,state,restored
        gc.collect();torch.cuda.empty_cache()
    ref=load_best('FR29')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):assert ref(x).shape==y.shape
    assert parameters(ref)==278804
    report['reference_checkpoint']='FR29 strict load and manifest/config/source/metric identity PASS; not retrained'
    report['status']='PASS';save(R/'smoke_summary.json',report)
    save(R/'status.json',{'status':'SMOKE_PASS','formal_training_started':False})

def train(name):
    assert json.loads((R/'smoke_summary.json').read_text(encoding='utf-8'))['status']=='PASS'
    out=CACHE/name;out.mkdir(exist_ok=True)
    runtime.seed_all(CFG['seed']);d=data();net=model(name,True).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=CFG['epochs'])
    start_epoch=1;best=-1.;last=out/'last.pt'
    history=[]
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);verify(ck,name)
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sch.load_state_dict(ck['scheduler']);runtime.restore_rng(ck['rng'])
        start_epoch=ck['epoch']+1;best=ck['best_mIoU']
        history=[json.loads(s) for s in (out/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if s] if (out/'epochs.jsonl').exists() else []
        assert [r['epoch'] for r in history]==list(range(1,len(history)+1))
        if len(history)==ck['epoch']-1:
            history.append(ck['record'])
            with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(ck['record'])+'\n')
        assert len(history)==ck['epoch'];del ck
    torch.cuda.empty_cache()
    for epoch in range(start_epoch,41):
        t=time.perf_counter();torch.cuda.reset_peak_memory_stats();values=[];parts=[]
        sampler=np.random.default_rng(CFG['seed']+1009*epoch);order=np.array(d.splits['train']);sampler.shuffle(order)
        lr=opt.param_groups[0]['lr']
        for pos in range(0,len(order),8):
            ids=order[pos:pos+8].tolist();x,y=d.batch(ids,sampler.random(len(ids))<.5)
            v,n,p=update(net,opt,x,y);values.append(v);parts.append(p)
        peak_train=torch.cuda.max_memory_allocated()/2**20
        sch.step();val=evaluate(net,d)
        step=int(next(iter(opt.state.values()))['step'].item());assert step==271*epoch
        if val['mIoU']>best:
            best=val['mIoU'];state=checkpoint(net,opt,sch,epoch,best,name)
            state.pop('optimizer');state.pop('scheduler');state.pop('rng');state['validation']=val
            ck_save(out/'best.pt',state);del state
        record={'epoch':epoch,'model':name,'seed':42,'global_step':step,'updates':271,'samples':2168,
            'mean_training_loss':float(np.mean(values)),'components':{k:float(np.mean([p[k] for p in parts])) for k in parts[0]},
            'lr_used':lr,'lr_after':opt.param_groups[0]['lr'],'validation':val,'best_mIoU':best,
            'seconds':time.perf_counter()-t,'train_peak_allocated_MiB':peak_train,
            'train_val_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'train_val_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20}
        state=checkpoint(net,opt,sch,epoch,best,name);state['record']=record;ck_save(last,state);del state
        with (out/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        history.append(record);print(json.dumps(record),flush=True)
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);verify(ck,name)
    chosen=max(history,key=lambda r:(r['validation']['mIoU'],-r['epoch']))
    assert chosen['epoch']==ck['epoch'] and chosen['validation']==ck['validation']
    assert len(history)==40 and ck['validation']['images']==542
    save(out/'result.json',{'model':name,'status':'COMPLETE','seed':42,'epochs':40,'best_epoch':ck['epoch'],
        'validation':ck['validation'],'inference_parameters':parameters(net),'training_parameters':parameters(net),
        'train_val_peak_allocated_MiB':max(r['train_val_peak_allocated_MiB'] for r in history),
        'train_val_peak_reserved_MiB':max(r['train_val_peak_reserved_MiB'] for r in history),
        'training_seconds':sum(r['seconds'] for r in history),'last5_mIoU':[r['validation']['mIoU'] for r in history[-5:]],
        'last5_loss':[r['mean_training_loss'] for r in history[-5:]],'convergence_ceiling_claim':False})
    fields=['epoch','mean_training_loss','lr_used','seconds','mIoU','glass_IoU','MAE','boundary_F1','non_glass_FPR','glass_FNR']
    with (out/'curves.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in history:w.writerow({k:r[k] if k in r else r['validation'][k] for k in fields})

@torch.inference_mode()
def profile(name):
    runtime.seed_all(42);net=load_best(name);x=torch.zeros(1,3,256,256,device='cuda')
    counts={'conv_MACs':0,'linear_MACs':0,'attention_MACs':0};handles=[]
    def hook(m,inputs,output):
        if isinstance(m,nn.Conv2d):counts['conv_MACs']+=output.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m,nn.Linear):counts['linear_MACs']+=output.numel()*m.in_features
        elif getattr(m,'is_coarse_attention_block',False):
            b,t,c=inputs[0].shape;counts['attention_MACs']+=2*b*t*t*c
    for m in net.modules():
        assert not isinstance(m,nn.ConvTranspose2d), 'Extend common FLOPs accounting before using this op'
        if isinstance(m,(nn.Conv2d,nn.Linear)) or getattr(m,'is_coarse_attention_block',False):handles.append(m.register_forward_hook(hook))
    with torch.autocast('cuda',dtype=torch.bfloat16):net(x)
    for h in handles:h.remove()
    torch.cuda.synchronize();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(60):net(x)
        torch.cuda.synchronize()
    result={'model':name,'parameters':parameters(net),**counts,'matrix_FLOPs_2MACs':2*sum(counts.values()),
        'scope':CFG['benchmark'],'baseline_allocated_MiB':base/2**20,
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'incremental_activation_MiB':(torch.cuda.max_memory_allocated()-base)/2**20,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,'process':'Fresh subprocess, one model, no optimizer or data cache'}
    if name=='FR29':assert result['matrix_FLOPs_2MACs']==674545664
    save(CACHE/name/'efficiency.json',result)
    if name!='FR29':ck_save(CACHE/name/'deployment.pt',{'model':{k:v.cpu() for k,v in net.state_dict().items()},'parameters':parameters(net),'config':CFG,'variant':name})
    print(json.dumps(result),flush=True)

@torch.inference_mode()
def benchmark():
    runtime.seed_all(42)
    names=['FR29']+[n for n in CFG['runnable'] if (CACHE/n/'result.json').exists()]
    gpu=lambda:subprocess.run(['nvidia-smi','--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    before=gpu();nets={n:load_best(n) for n in names};times={n:[] for n in names}
    x=torch.zeros(1,3,256,256,device='cuda');orders=[names,names[::-1],names[::-1],names]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for net in nets.values():
            for _ in range(60):net(x)
        for order in orders:
            for _ in range(40):
                for name in order:
                    torch.cuda.synchronize();t=time.perf_counter();nets[name](x);torch.cuda.synchronize()
                    times[name].append(1000*(time.perf_counter()-t))
    save(R/'matched_latency.json',{'status':'COMPLETE','protocol':CFG['benchmark'],'gpu_before':before,'gpu_after':gpu(),
        'orders':orders,'samples_per_model':160,'results':{n:{'median_ms':float(np.median(t)),'p95_ms':float(np.percentile(t,95)),
        'round_medians_ms':[float(np.median(t[i:i+40])) for i in range(0,160,40)]} for n,t in times.items()},
        'raw_ms':times,'limitation':'Shared laptop GPU; clocks/power not locked; low external load audited, no cross-device latency claim'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','train','profile','benchmark']);p.add_argument('--model');a=p.parse_args()
    check_sources();runtime.seed_all(42)
    {'smoke':smoke,'train':lambda:train(a.model),'profile':lambda:profile(a.model),'benchmark':benchmark}[a.mode]()
