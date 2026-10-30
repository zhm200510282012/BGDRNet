"""One locked model/dataset evaluation. No fitting, calibration, or prediction-map writes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,csv,gc,importlib,io,json,sys,time,zipfile
import numpy as np
from PIL import Image
import torch
import cv2
import evaluation_helpers as h
R=Path(__file__).resolve().parent

def load(name):
    index=json.loads((R/'checkpoints.frozen.json').read_text(encoding='utf-8'));entry=index['models'][name]
    root=Path(entry['source']);assert all((root/k).read_text(encoding='utf-8')==v for k,v in entry['sources'].items())
    sys.path.insert(0,str(root));runtime=importlib.import_module('runtime');runtime.seed_all(42)
    if entry['kind']=='external':
        module=importlib.import_module('models');net=module.build(name)
        make_training=lambda:module.build(name)
    else:
        module=importlib.import_module('model');net=module.TinyGlass(entry['variant'],training_heads=False)
        make_training=lambda:module.TinyGlass(entry['variant'],training_heads=True)
    ck=torch.load(entry['best'],map_location='cpu',weights_only=False)
    assert ck['epoch']==entry['best_epoch'] and ck['validation']==entry['validation']
    assert ck['config']['seed']==entry['seed'] and ck['variant']==entry['variant']
    assert ck['source_text']==entry['sources']
    deploy=torch.load(entry['deployment'],map_location='cpu',weights_only=False)
    net.load_state_dict(deploy['model'],strict=True)
    assert all(torch.equal(v,ck['model'][k]) for k,v in net.state_dict().items())
    assert sum(p.numel() for p in net.parameters())==entry['parameters']
    full=make_training();full.load_state_dict(ck['model'],strict=True)
    del ck,deploy
    return net.cuda().eval(),full.cuda().eval(),runtime,entry

def pairs(dataset,rows):
    if dataset=='GEGD2':
        with zipfile.ZipFile('D:/DSP/GEGD2.zip') as z:
            for row in rows:
                yield h.read_pair(z,row)
    else:
        for row in rows:
            with Image.open(row['image_path']) as im:x=np.asarray(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)).copy()
            with Image.open(row['mask_path']) as im:truth=np.asarray(im)>0
            assert truth.ndim==2
            yield x,truth

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',choices=['GSD_test','GEGD2'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2)
    f=json.loads((R/'execution.frozen.json').read_text(encoding='utf-8'))
    assert all((R/k).read_text(encoding='utf-8')==v for k,v in f['sources'].items())
    assert all((R/k).read_text(encoding='utf-8')==v for k,v in f['test_manifests'].items())
    mf=R/('gsd_test_manifest.frozen.csv' if a.dataset=='GSD_test' else 'gegd2_manifest.frozen.csv')
    manifest=list(csv.DictReader(mf.open(encoding='utf-8-sig')));assert len(manifest)==(813 if a.dataset=='GSD_test' else 950)
    out=R/'evaluation'/a.dataset/a.model;out.mkdir(parents=True,exist_ok=True)
    net,full,runtime,entry=load(a.model)
    x,truth=next(pairs(a.dataset,manifest[:1]));z=h.forward(net,x);z2=h.forward(full,x);assert np.array_equal(z,z2)
    measured=h.measure(z,truth,runtime.boundary_f1)
    assert measured['TP']+measured['FP']+measured['TN']+measured['FN']==truth.size
    if a.smoke:
        h.write_json(out/'smoke.json',{'status':'PASS','model':a.model,'dataset':a.dataset,
            'strict_checkpoint_source_and_deployment_checked':True,'real_input_best_deployment_exact':True,
            'output_shape':[1,1,256,256],'label_rule':'raw==255' if a.dataset=='GEGD2' else 'raw>0',
            'parameters':entry['parameters'],'checkpoint_epoch':entry['best_epoch'],'checkpoint_index_frozen_before_first_evaluation':True})
        print(json.dumps({'event':'evaluation_smoke_pass','model':a.model,'dataset':a.dataset}),flush=True);return
    assert json.loads((out/'smoke.json').read_text(encoding='utf-8'))['status']=='PASS'
    del full,x,truth,z,z2,measured;gc.collect();torch.cuda.empty_cache()
    done=[]
    for path in sorted(out.glob('part_*.csv')):done.extend(csv.DictReader(path.open(encoding='utf-8')))
    assert [r['sample_id'] for r in done]==[r['sample_id'] for r in manifest[:len(done)]]
    t=time.perf_counter();pending=[];first=len(done)
    for i,(row,(x,truth)) in enumerate(zip(manifest[first:],pairs(a.dataset,manifest[first:])),start=first):
        z=h.forward(net,x);m=h.measure(z,truth,runtime.boundary_f1)
        pending.append({'sample_id':row['sample_id'],**m})
        del x,truth,z,m
        if len(pending)==100 or i==len(manifest)-1:
            h.write_chunk(out,pending,len(done));done.extend(pending);pending=[]
            h.write_json(out/'status.json',{'status':'RUNNING','completed_images':len(done),'total':len(manifest)})
            print(json.dumps({'model':a.model,'dataset':a.dataset,'completed':len(done),'total':len(manifest)}),flush=True)
    result={'status':'COMPLETE','model':a.model,'dataset':a.dataset,'checkpoint_epoch':entry['best_epoch'],
        'metrics':h.aggregate(done),'seconds_this_invocation':time.perf_counter()-t,'resumed_images':first,
        'threshold':.5,'input_size':256,'precision':'CUDA BF16 autocast, FP32 weights','batch_size':1,
        'source_training':'GSD training split only; no target adaptation','evaluation_history_disclosed':True,
        'gt_rule':'raw==255' if a.dataset=='GEGD2' else 'raw>0','test_tuning':False}
    h.write_json(out/'result.json',result);h.write_json(out/'status.json',{'status':'COMPLETE','images':len(done)})
    print(json.dumps({'event':'evaluation_complete',**result}),flush=True)
if __name__=='__main__':main()
