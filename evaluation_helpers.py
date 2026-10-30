"""Frozen GSD checkpoints -> GEGD2; no training, tuning, plots, or prediction maps."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse, csv, gc, importlib, inspect, io, json, platform, sys, time, zipfile
import numpy as np
from PIL import Image
import cv2
import torch

ROOT=Path(__file__).resolve().parent
AUDIT=Path('D:/SIVP/glass_independent_data_audit_20261027')

def write_json(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); temp.replace(path)

def frozen_json(path,obj):
    if path.exists(): assert json.loads(path.read_text(encoding='utf-8'))==obj, f'Frozen file changed: {path}'
    else: write_json(path,obj)

def aggregate(rows):
    tp,fp,tn,fn=[sum(int(r[k]) for r in rows) for k in ['TP','FP','TN','FN']]
    pixels=tp+fp+tn+fn
    return {'images':len(rows),'pixels':pixels,'TP':tp,'FP':fp,'TN':tn,'FN':fn,
      'mIoU':(tp/(tp+fp+fn)+tn/(tn+fp+fn))/2,'glass_IoU':tp/(tp+fp+fn),
      'background_IoU':tn/(tn+fp+fn),'non_glass_FPR':fp/(fp+tn),'glass_FNR':fn/(fn+tp),
      'MAE':sum(float(r['absolute_error_sum']) for r in rows)/pixels,
      'boundary_F1':sum(float(r['boundary_F1']) for r in rows)/len(rows),
      'predicted_foreground_fraction':(tp+fp)/pixels,'true_foreground_fraction':(tp+fn)/pixels}

def load_setup(variant,out):
    index=json.loads((AUDIT/'frozen_checkpoint_index.json').read_text(encoding='utf-8'))
    entry=next(r for r in index if r['id']==variant+':42')
    source=Path(entry['result_source']).parent.parent.parent
    sys.path.insert(0,str(source))
    module=importlib.import_module('model'); runtime=importlib.import_module('runtime')
    sources={p.name:p.read_text(encoding='utf-8-sig') for p in sorted(source.glob('*.py'))}
    frozen_json(out/'sources.frozen.json',sources)
    reference=(Path('D:/SIVP/glass_fr12_lowrank_entry_20261026/jobs/FR21/runtime.py')).read_text(encoding='utf-8-sig')
    assert inspect.getsource(runtime.boundary_f1).strip() in reference
    torch.set_num_threads(4); torch.set_num_interop_threads(1); cv2.setNumThreads(4)
    runtime.seed_all(42)
    best=torch.load(entry['files']['best.pt']['path'],map_location='cpu',weights_only=False)
    deploy=torch.load(entry['files']['deployment.pt']['path'],map_location='cpu',weights_only=False)
    assert best['variant']==variant and deploy['variant']==variant
    assert best['epoch']==entry['source_development_best_epoch']
    model=module.TinyGlass(variant,weights=None,training_heads=False)
    model.load_state_dict(deploy['model'],strict=True)
    assert all(k in best['model'] and torch.equal(v,best['model'][k]) for k,v in model.state_dict().items())
    params=sum(p.numel() for p in model.parameters()); assert params==entry['parameters']
    train_model=module.TinyGlass(variant,weights=None,training_heads=True)
    train_model.load_state_dict(best['model'],strict=True)
    model=model.cuda().eval(); train_model=train_model.cuda().eval()
    evidence={'variant':variant,'source':str(source),'parameters':params,'best_epoch':best['epoch'],
      'strict_load':True,'all_deployment_tensors_equal_best':True,
      'deployment_checkpoint':entry['files']['deployment.pt']['path'],
      'best_checkpoint':entry['files']['best.pt']['path'],
      'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
      'numpy':np.__version__,'opencv':cv2.__version__,'gpu':torch.cuda.get_device_name(),
      'threads':4,'opencv_threads':4,'precision':'CUDA BF16 autocast; FP32 weights',
      'batch_size':1,'input_size':256,'threshold':0.5,'gt_rule':'uint8 == 255'}
    del best,deploy
    return model,train_model,runtime,evidence

def read_pair(z,row):
    with Image.open(io.BytesIO(z.read(row['image_member']))) as im:
        assert im.size==(int(row['width']),int(row['height']))
        x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
    with Image.open(io.BytesIO(z.read(row['glass_member']))) as im:
        raw=np.asarray(im); truth=raw==255
    assert truth.shape==(int(row['height']),int(row['width']))
    assert int(truth.sum())==int(row['glass_eq255_pixels'])
    return x,truth

@torch.inference_mode()
def forward(model,x):
    tensor=torch.from_numpy(x).permute(2,0,1)[None].contiguous().cuda().float().div_(255.)
    with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(tensor)
    assert logits.shape==(1,1,256,256) and bool(torch.isfinite(logits).all())
    return logits[0,0].float().cpu().numpy()

def measure(logits,truth,boundary_f1):
    logit=cv2.resize(logits,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_LINEAR)
    prob=1/(1+np.exp(-np.clip(logit,-80,80))); del logit
    pred=prob>=.5
    tp=int(np.count_nonzero(pred & truth)); ng=int(np.count_nonzero(truth)); npred=int(np.count_nonzero(pred))
    fn=ng-tp; fp=npred-tp; tn=truth.size-tp-fp-fn
    ae=float(np.abs(prob-truth).sum(dtype=np.float64)); del prob
    bf=boundary_f1(pred,truth)
    assert tp+fp+tn+fn==truth.size and np.isfinite(ae) and 0<=bf<=1
    return {'TP':tp,'FP':fp,'TN':tn,'FN':fn,'pixels':truth.size,'absolute_error_sum':ae,'boundary_F1':bf}

def write_chunk(out,rows,start):
    path=out/f'part_{start:04d}.csv'; assert not path.exists()
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    temp.replace(path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--variant',required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args(); out=ROOT/'runs'/args.variant;out.mkdir(parents=True,exist_ok=True)
    with (AUDIT/'target_manifest.frozen.csv').open(encoding='utf-8-sig') as f: manifest=list(csv.DictReader(f))
    assert len(manifest)==950 and len({r['sample_id'] for r in manifest})==950
    frozen_json(out/'target_manifest.frozen.json',manifest)
    model,trained,runtime,evidence=load_setup(args.variant,out)
    with zipfile.ZipFile('D:/DSP/GEGD2.zip') as z:
        # Fixed input-content-independent smoke choices: first, largest, first gray-valued mask.
        smoke_ids=list(dict.fromkeys([0,max(range(len(manifest)),key=lambda i:int(manifest[i]['pixels'])),
                    next(i for i,r in enumerate(manifest) if int(r['intermediate_gray_pixels'])>0)]))
        if args.smoke:
            timings=[]; start=time.perf_counter()
            for i in smoke_ids:
                t=time.perf_counter(); x,truth=read_pair(z,manifest[i]); a=forward(model,x); b=forward(trained,x)
                assert np.array_equal(a,b), 'Best/deployment real-input mismatch'
                measured=measure(a,truth,runtime.boundary_f1)
                # Perfect/inverted-label sanity checks using this real mask; no extra model evaluation.
                assert measured['TP']+measured['FN']==int(manifest[i]['glass_eq255_pixels'])
                timings.append({'sample_id':manifest[i]['sample_id'],'pixels':truth.size,'seconds':time.perf_counter()-t})
                del truth,x,a,b,measured
            evidence.update(status='SMOKE_PASS',real_images=len(smoke_ids),best_deploy_exact_output=True,
               timings=timings,seconds=time.perf_counter()-start,
               full_run_estimate_seconds=sum(t['seconds'] for t in timings)/sum(t['pixels'] for t in timings)*sum(int(r['pixels']) for r in manifest))
            write_json(out/'smoke.json',evidence)
            print(json.dumps({k:evidence[k] for k in ['variant','status','parameters','real_images','seconds','full_run_estimate_seconds']}),flush=True)
            return
        assert (out/'smoke.json').exists() and json.loads((out/'smoke.json').read_text())['status']=='SMOKE_PASS'
        # Use the first frozen sample only to recheck deployment fidelity, without choosing anything.
        x,_=read_pair(z,manifest[0]); assert np.array_equal(forward(model,x),forward(trained,x))
        del trained,x;gc.collect();torch.cuda.empty_cache()
        rows=[]
        for part in sorted(out.glob('part_*.csv')):
            with part.open(encoding='utf-8') as f: rows.extend(csv.DictReader(f))
        assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in manifest[:len(rows)]]
        start=time.perf_counter(); pending=[]; first=len(rows)
        write_json(out/'status.json',{'status':'RUNNING','completed_images':first,'total':950})
        for i in range(first,len(manifest)):
            t=time.perf_counter(); row=manifest[i]; x,truth=read_pair(z,row); logits=forward(model,x)
            metrics=measure(logits,truth,runtime.boundary_f1)
            pending.append({'sample_id':row['sample_id'],'domain_inferred':row['domain_inferred'],**metrics,
                           'evaluation_seconds':time.perf_counter()-t})
            del x,truth,logits,metrics
            if len(pending)==25 or i==len(manifest)-1:
                write_chunk(out,pending,len(rows)); rows.extend(pending);pending=[]
                write_json(out/'status.json',{'status':'RUNNING','completed_images':len(rows),'total':950})
                print(json.dumps({'variant':args.variant,'completed_images':len(rows),'total':950}),flush=True)
        result={'status':'COMPLETE','variant':args.variant,'source_training':'GSD only; seed42',
                'target':'all 950 official GEGD2 test images','metrics':aggregate(rows),'setup':evidence,
                'domain_metrics_INFERRED_NOT_AUTHOR_VERIFIED':{d:aggregate([r for r in rows if r['domain_inferred']==d]) for d in sorted({r['domain_inferred'] for r in rows})},
                'this_invocation_seconds':time.perf_counter()-start,'resumed_from_images':first,
                'no_training':True,'no_test_tuning':True,'no_scientific_figures':True}
        write_json(out/'result.json',result)
        write_json(out/'status.json',{'status':'COMPLETE','completed_images':950,'total':950})
        print(json.dumps({'variant':args.variant,'status':'COMPLETE','metrics':result['metrics']}),flush=True)

if __name__=='__main__': main()
