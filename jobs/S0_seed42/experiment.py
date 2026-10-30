import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from model import TinyGlass,count_parameters,loss_bundle
import runtime
BASE=Path(__file__).resolve().parent
CFG=json.loads((BASE/'config.frozen.json').read_text(encoding='utf-8'))
MANIFEST=(BASE/'manifest.csv').read_text(encoding='utf-8')
SOURCES={k:(BASE/k).read_text(encoding='utf-8') for k in ['model.py','runtime.py','experiment.py','worker.py']}
DEVICE=torch.device('cuda')


def save_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temp,path)

def validate_manifest():
    with (BASE/'manifest.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    assert len(rows)==2710
    assert sum(r['split']=='train' for r in rows)==2168 and sum(r['split']=='val' for r in rows)==542
    assert len({r['sample_id'] for r in rows})==2710
    for row in rows:
        assert row['split'] in ['train','val'] and row['official_split']=='train'
        for key in ['image_path','mask_path']:
            p=Path(row[key]); assert p.is_file() and p.stat().st_size>0
            assert 'test' not in [part.lower() for part in p.parts]
    save_json(BASE/'logs/data_access_contract.json',{'status':'PASS','counts':{'train':2168,'val':542},
        'source':'prior audited GSD official training split; identical train/val membership',
        'official_test_rows_in_manifest':0,'test_images_or_masks_opened_by_this_runner':False,
        'scene_grouping':'unavailable; image-level split only','data_content_reaudit':'previous paired decode audit retained; all required files rechecked for existence and nonzero size'})

def data_cache():
    validate_manifest()
    data=runtime.DataCache(CFG['size'])
    assert not data.splits['test']
    data.splits.pop('test')
    return data

def evaluate(model,data):
    assert set(data.splits)=={'train','val'}
    result=runtime.evaluate(model,data,'val',CFG['batch_size'])
    result['glass_FNR']=result['FN']/(result['TP']+result['FN'])
    return result

def make_checkpoint(model,optimizer,scheduler,epoch,best):
    return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch':epoch,'best_mIoU':best,'variant':model.variant,'config':CFG,'manifest':MANIFEST,
            'rng':runtime.rng_state(),'source_text':SOURCES}

def save_checkpoint(path,state):
    temp=path.with_suffix('.tmp'); torch.save(state,temp); os.replace(temp,path)

def verify_checkpoint(state,variant):
    assert state['config']==CFG and state['manifest']==MANIFEST and state['variant']==variant
    assert state['source_text']==SOURCES, 'Source changed; mixed-source resume is rejected'

def checked_update(model,optimizer,x,y):
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss,meta=loss_bundle(model,x,y,CFG)
    loss.backward()
    norm=nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()),float(norm),meta

@torch.inference_mode()
def profile(variant):
    runtime.seed_all(CFG['seed'])
    checkpoint=torch.load(BASE/'runs'/variant/'best.pt',map_location='cpu',weights_only=False)
    verify_checkpoint(checkpoint,variant)
    model=TinyGlass(variant,training_heads=True); model.load_state_dict(checkpoint['model'])
    deployed=model.deployment_copy().to(DEVICE).eval(); del model,checkpoint
    x=torch.zeros(1,3,CFG['size'],CFG['size'],device=DEVICE)
    counts={'conv':0,'linear':0,'attention':0}; handles=[]
    def hook(module,inputs,output):
        if isinstance(module,nn.Conv2d):
            counts['conv']+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module,nn.Linear):
            counts['linear']+=output.numel()*module.in_features
        elif getattr(module,'is_coarse_attention_block',False):
            batch,tokens,channels=inputs[0].shape
            counts['attention']+=2*batch*tokens*tokens*channels
    for module in deployed.modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)) or getattr(module,'is_coarse_attention_block',False): handles.append(module.register_forward_hook(hook))
    with torch.autocast('cuda',dtype=torch.bfloat16): deployed(x)
    for handle in handles: handle.remove()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    baseline=torch.cuda.memory_allocated()
    for _ in range(50):
        with torch.autocast('cuda',dtype=torch.bfloat16): deployed(x)
    times=[]
    for _ in range(200):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.autocast('cuda',dtype=torch.bfloat16): deployed(x)
        end.record(); end.synchronize(); times.append(start.elapsed_time(end))
    result={'parameters':count_parameters(deployed),'conv_MACs':counts['conv'],'conv_FLOPs_2MACs':2*counts['conv'],
        'linear_MACs':counts['linear'],'attention_MACs':counts['attention'],'matrix_FLOPs_2MACs':2*sum(counts.values()),
        'matrix_FLOPs_scope':'conv + linear + QK/AV matrix operations; excludes normalization, softmax, interpolation and elementwise ops',
        'FLOPs_scope':'conv only; excludes normalization, bias, pooling, activations, resize and elementwise operations',
        'latency_ms_median':float(np.median(times)),'latency_ms_p95':float(np.percentile(times,95)),
        'precision':'AMP BF16, FP32 weights; batch1; model only; no disk/CPU postprocess',
        'measurement_process':'fresh subprocess for each variant; no optimizer or data cache in this process',
        'inference_baseline_allocated_MiB':baseline/2**20,
        'inference_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'inference_incremental_allocated_MiB':(torch.cuda.max_memory_allocated()-baseline)/2**20,
        'inference_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20}
    assert CFG.get('inference_parameter_ceiling') is None or result['parameters']<=CFG['inference_parameter_ceiling']
    save_checkpoint(BASE/'runs'/variant/'deployment.pt',{'model':{k:v.detach().cpu() for k,v in deployed.state_dict().items()},
        'variant':variant,'config':CFG,'parameters':result['parameters'],'training_heads_removed':True})
    save_json(BASE/'runs'/variant/'efficiency.json',result)
    print(json.dumps(result),flush=True)

def train_one(variant,data):
    out=BASE/'runs'/variant; out.mkdir(parents=True,exist_ok=True)
    runtime.seed_all(CFG['seed'])
    model=TinyGlass(variant,CFG['weights'],training_heads=True).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG['epochs'])
    first=1; best=-1.; last=out/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location='cpu',weights_only=False); verify_checkpoint(saved,variant)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        runtime.restore_rng(saved['rng']); first=saved['epoch']+1; best=saved['best_mIoU']
        logpath=out/'epochs.jsonl'
        logrows=[json.loads(s) for s in logpath.read_text(encoding='utf-8').splitlines() if s.strip()] if logpath.exists() else []
        assert [r['epoch'] for r in logrows]==list(range(1,len(logrows)+1)), 'Noncontiguous epoch log'
        if len(logrows)==saved['epoch']-1:
            assert saved['last_epoch_record']['epoch']==saved['epoch']
            with logpath.open('a',encoding='utf-8') as f:f.write(json.dumps(saved['last_epoch_record'])+'\n')
        else:assert len(logrows)==saved['epoch'], 'Checkpoint/log mismatch beyond single atomic-save window'
        del saved
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for epoch in range(first,CFG['epochs']+1):
        start=time.monotonic(); losses=[]; components=[]; model.train()
        lr_used=[g['lr'] for g in optimizer.param_groups]; sample_count=0; update_count=0
        sampler=np.random.default_rng(CFG['seed']+1009*epoch)
        order=np.array(data.splits['train']); sampler.shuffle(order)
        for pos in range(0,len(order),CFG['batch_size']):
            ids=order[pos:pos+CFG['batch_size']].tolist(); x,y=data.batch(ids,sampler.random(len(ids))<.5)
            loss,norm,meta=checked_update(model,optimizer,x,y); losses.append(loss); components.append(meta)
            sample_count+=len(ids); update_count+=1
        train_peak=torch.cuda.max_memory_allocated()/2**20
        scheduler.step(); validation=evaluate(model,data)
        if validation['mIoU']>best:
            best=validation['mIoU']
            save_checkpoint(out/'best.pt',{'model':model.state_dict(),'epoch':epoch,'validation':validation,
                'variant':variant,'config':CFG,'manifest':MANIFEST,'source_text':SOURCES})
        record={'timestamp':runtime.stamp(),'variant':variant,'seed':CFG['seed'],'epoch':epoch,
            'mean_training_loss':float(np.mean(losses)),
            'lr_used':lr_used,'lr_after':[g['lr'] for g in optimizer.param_groups],
            'global_step':int(next(iter(optimizer.state.values()))['step'].item()),
            'samples_seen_this_epoch':sample_count,'updates_this_epoch':update_count,
            'components':{k:float(np.mean([m[k] for m in components])) for k in ['bce','iou_loss','region_loss','edge_loss']},
            'validation':validation,'best_mIoU':best,'seconds':time.monotonic()-start,
            'training_and_validation_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
            'training_peak_before_current_validation_MiB':train_peak,
            'training_and_validation_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20}
        state=make_checkpoint(model,optimizer,scheduler,epoch,best);state['last_epoch_record']=record
        save_checkpoint(last,state);del state
        with (out/'epochs.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    model.eval(); ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    history=[json.loads(line) for line in (out/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if line]
    result={'variant':variant,'seed':CFG['seed'],'best_epoch':ck['epoch'],'validation':ck['validation'],
        'training_parameters':count_parameters(model),
        'train_val_peak_allocated_MiB':max(row['training_and_validation_peak_allocated_MiB'] for row in history),
        'train_val_peak_reserved_MiB':max(row['training_and_validation_peak_reserved_MiB'] for row in history)}
    del model,optimizer,scheduler,ck
    if 'x' in locals(): del x,y
    torch.cuda.empty_cache()
    # A one-shot subprocess runs to completion. This is not a monitoring/polling loop.
    measured=subprocess.run([sys.executable,'-u',str(BASE/'experiment.py'),'--mode','profile','--variant',variant],
        capture_output=True,text=True,encoding='utf-8')
    (out/'profile.stdout.log').write_text(measured.stdout,encoding='utf-8')
    (out/'profile.stderr.log').write_text(measured.stderr,encoding='utf-8')
    if measured.returncode: raise RuntimeError('Fresh-process profile failed: '+measured.stderr)
    result['efficiency']=json.loads((out/'efficiency.json').read_text(encoding='utf-8'))
    save_json(out/'result.json',result)
    return result
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['profile'],required=True);a.add_argument('--variant',choices=['S0'],required=True);args=a.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2);runtime.seed_all(CFG['seed']);profile(args.variant)
