import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime,timezone
import csv,gc,importlib.util,io,json,sys,time,zipfile
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
import cv2
R=Path(__file__).resolve().parent
CORE=Path('D:/SIVP/glass_fr29_factorial_evidence_20261029')
CACHE=Path('D:/大创/cache/glass_task_baseline_20261029')
sys.path.insert(0,str(R/'author_code'))
from model import GlassNet
from data_loader import RescaleT,ToTensorLab
sys.path.append(str(CORE))
import evaluation_helpers as h

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,o):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def freeze():
    files=list((R/'author_code').rglob('*.py'))+[R/'run_reference.py',R/'PROTOCOL.md']
    record={'sources':{str(p.relative_to(R)):p.read_text(encoding='utf-8-sig') for p in files},
        'shared_metric_helpers':(CORE/'evaluation_helpers.py').read_text(encoding='utf-8'),
        'manifests':{n:(CORE/n).read_text(encoding='utf-8') for n in ['gsd_test_manifest.frozen.csv','gegd2_manifest.frozen.csv']},
        'weight':str(CACHE/'GSD.pth'),'bytes':(CACHE/'GSD.pth').stat().st_size,
        'revision':'71607d06ebdcc35e0b2cf92620ca657e74eb3ee6','condition':'GlassNet_published_standard256_noCRF','threshold':.5}
    assert record['bytes']==335619863
    p=R/'execution.frozen.json'
    if p.exists():assert read(p)==record
    else:save(p,record)

def preprocess(image):
    # These author transforms do not read real labels or reflections.
    sample={'image':image,'label':np.zeros((1,1,1),np.uint8),'reflection':np.zeros((1,1,3),np.uint8)}
    sample=RescaleT(256)(sample);out=ToTensorLab(flag=0)(sample)['image']
    return out[None].contiguous().cuda().float()

@torch.inference_mode()
def forward(net,image):
    x=preprocess(image)
    with torch.autocast('cuda',dtype=torch.bfloat16):out=net(x)[0]
    assert out.shape==(1,1,256,256) and torch.isfinite(out).all()
    return out[0,0].float().cpu().numpy()

def pairs(dataset,rows):
    if dataset=='GEGD2':
        with zipfile.ZipFile('D:/DSP/GEGD2.zip') as z:
            for row in rows:
                with Image.open(io.BytesIO(z.read(row['image_member']))) as im:im=np.asarray(im.convert('RGB')).copy()
                with Image.open(io.BytesIO(z.read(row['glass_member']))) as mask:truth=np.asarray(mask)==255
                assert int(truth.sum())==int(row['glass_eq255_pixels'])
                yield im,truth
    else:
        for row in rows:
            with Image.open(row['image_path']) as im:im=np.asarray(im.convert('RGB')).copy()
            with Image.open(row['mask_path']) as mask:truth=np.asarray(mask)>0
            yield im,truth

@torch.inference_mode()
def profile(net):
    raw=np.asarray(Image.open('D:/DSP/GSD/train/image/-fwmf0AkK7U.jpg').convert('RGB'));x=preprocess(raw)
    counts={'conv_MACs':0,'linear_MACs':0};handles=[]
    def hook(m,inp,out):
        if isinstance(m,nn.Conv2d):counts['conv_MACs']+=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
        elif isinstance(m,nn.Linear):counts['linear_MACs']+=out.numel()*m.in_features
    for m in net.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):handles.append(m.register_forward_hook(hook))
    with torch.autocast('cuda',dtype=torch.bfloat16):net(x)
    for hook_handle in handles:hook_handle.remove()
    torch.cuda.synchronize();torch.cuda.empty_cache();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for _ in range(60):net(x)
    torch.cuda.synchronize()
    result={'parameters':sum(p.numel() for p in net.parameters()),**counts,'matrix_FLOPs_2MACs':2*sum(counts.values()),
        'single_model_baseline_allocated_MiB':base/2**20,'single_model_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'single_model_incremental_forward_MiB':(torch.cuda.max_memory_allocated()-base)/2**20,
        'single_model_peak_reserved_MiB':torch.cuda.max_memory_reserved()/2**20,
        'scope':'All original model parameters/outputs retained, no optimizer; matrix ops only; model forward excludes external author CPU transforms'}
    # Load the frozen FR29 canonical module without replacing the author model module.
    shared=Path(read(CORE/'execution.frozen.json')['jobs']['RAW_SHARED']['path']);sys.path.insert(0,str(shared))
    spec=importlib.util.spec_from_file_location('fr29_for_pair',shared/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    entry=read(CORE/'checkpoints.frozen.json')['models']['FR29'];ref=mod.TinyGlass('FR29',training_heads=False).cuda().eval()
    ck=torch.load(entry['deployment'],map_location='cpu',weights_only=False);ref.load_state_dict(ck['model'],strict=True);del ck
    rgb=torch.from_numpy(np.asarray(Image.fromarray(raw).resize((256,256),Image.Resampling.BILINEAR)).copy()).permute(2,0,1)[None].contiguous().cuda().float()/255
    nets={'GlassNet_published_standard256_noCRF':(net,x),'FR29':(ref,rgb)};names=list(nets);times={n:[] for n in names}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for m,inp in nets.values():
            for _ in range(60):m(inp)
        for order in [names,names[::-1],names[::-1],names]:
            for _ in range(40):
                for n in order:
                    m,inp=nets[n];torch.cuda.synchronize();t=time.perf_counter();m(inp);torch.cuda.synchronize();times[n].append(1000*(time.perf_counter()-t))
    result['paired_latency']={n:{'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))} for n,v in times.items()}
    result['precision']='SameGPU /256 /batch1 /BF16 autocast /FP32 weights; synchronized host,160samples';save(R/'efficiency.json',result)
    del ref,rgb,x;gc.collect();torch.cuda.empty_cache()

def report(results):
    lines=['# GlassNet作者权重：共同输出协议的补充参考','',
        '该参考为原网络+作者发布权重、256输入、原RGB归一化及本项目统一后处理；不使用作者预测min-max/CRF。完整作者384+CRF流程因pydensecrf缺失记BLOCKED，不是完整官方复现。训练样本清单/预训练/预算与FR29不匹配，不能并入同预算模型排名。',
        '', '| 数据集 | 方法 | mIoU% | glassIoU% | BF1% | FPR% | FNR% | MAE |','|---|---|---:|---:|---:|---:|---:|---:|']
    for d,r in results.items():
        other=read(CORE/'evaluation'/d/'FR29/result.json')['metrics']
        for n,v in [('FR29',other),('GlassNet_published_standard256_noCRF',r['metrics'])]:
            lines.append('| '+d+' | '+n+' | '+' | '.join([f"{100*v[k]:.3f}" for k in ['mIoU','glass_IoU','boundary_F1','non_glass_FPR','glass_FNR']]+[f"{v['MAE']:.5f}"] )+' |')
    if (R/'efficiency.json').exists():
        e=read(R/'efficiency.json');lines.extend(['',f"完整原模型参数{e['parameters']}，矩阵GFLOPs{e['matrix_FLOPs_2MACs']/1e9:.3f}；模型内全部旁路/反射输出保留。",''])
        for n,v in e['paired_latency'].items():lines.append(f"- 本场{n} median {v['median_ms']:.3f}ms，p95 {v['p95_ms']:.3f}ms。")
    lines.extend(['','GSD/GEGD2都有历史观察，当前模型与协议先冻结后评估，无调参或主模型切换。作者权重训练于GSD、没有GEGD2目标适配的直接证据；若发布样本列表缺失，跨库训练来源独立性仍有限制。不得将原文公开分数和本表相减，不以本表宣称完整GlassNet被超越。未绘图。'])
    (R/'FEASIBILITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    assert read(CORE/'status.json')['status']=='COMPLETE','Core pipeline must finish and release GPU before this reference'
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2)
    torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    freeze();save(R/'status.json',{'status':'LOADING_AUTHOR_WEIGHTS','full_author_recipe':'BLOCKED_PYDENSECRF'})
    state=torch.load(CACHE/'GSD.pth',map_location='cpu',weights_only=True);net=GlassNet();net.load_state_dict(state,strict=True);del state
    trainrow=next(r for r in csv.DictReader((Path('D:/大创/cache/glass_B40_FR29_20261028/FR29')/'manifest.csv').open(encoding='utf-8')) if r['split']=='train')
    with Image.open(trainrow['image_path']) as im:raw=np.asarray(im.convert('RGB')).copy()
    net=net.cuda().eval();before=forward(net,raw)
    path=CACHE/'restore_smoke.pt';torch.save(net.cpu().state_dict(),path);net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));net.cuda().eval()
    assert np.array_equal(before,forward(net,raw));del raw,before;gc.collect();torch.cuda.empty_cache()
    save(R/'smoke.json',{'status':'PASS','strict_full_weights':True,'output_shape':[1,1,256,256],
        'source_training_image':trainrow['sample_id'],'save_restore_output_exact':True,'parameters':sum(p.numel() for p in net.parameters()),
        'real_test_access_before_smoke':False,'no_fitting':True})
    if not (R/'efficiency.json').exists():profile(net)
    runtime_path=Path(read(CORE/'execution.frozen.json')['jobs']['RAW_SHARED']['path'])/'runtime.py'
    spec=importlib.util.spec_from_file_location('common_metrics_runtime',runtime_path);metrics_runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics_runtime)
    results={}
    for dataset,filename in [('GSD_test','gsd_test_manifest.frozen.csv'),('GEGD2','gegd2_manifest.frozen.csv')]:
        out=R/'evaluation'/dataset;out.mkdir(parents=True,exist_ok=True)
        if (out/'result.json').exists():results[dataset]=read(out/'result.json');continue
        rows=list(csv.DictReader((CORE/filename).open(encoding='utf-8-sig')));done=[]
        for p in sorted(out.glob('part_*.csv')):done.extend(csv.DictReader(p.open(encoding='utf-8')))
        assert [r['sample_id'] for r in done]==[r['sample_id'] for r in rows[:len(done)]]
        pending=[];first=len(done)
        for i,(row,(image,truth)) in enumerate(zip(rows[first:],pairs(dataset,rows[first:])),start=first):
            z=forward(net,image);m=h.measure(z,truth,metrics_runtime.boundary_f1)
            pending.append({'sample_id':row['sample_id'],**m});del image,truth,z,m
            if len(pending)==100 or i==len(rows)-1:
                h.write_chunk(out,pending,len(done));done.extend(pending);pending=[]
                print(json.dumps({'dataset':dataset,'completed':len(done),'total':len(rows)}),flush=True)
        results[dataset]={'status':'COMPLETE','condition':'GlassNet_published_standard256_noCRF','metrics':h.aggregate(done),'threshold':.5,'no_test_tuning':True}
        save(out/'result.json',results[dataset]);report(results)
    save(R/'summary.json',results);save(R/'status.json',{'status':'COMPLETE_ADAPTED_AUTHOR_WEIGHT_REFERENCE','full_author_recipe':'BLOCKED_PYDENSECRF','no_fitting':True});report(results)
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        save(R/'status.json',{'status':'STOPPED_ERROR','error':traceback.format_exc()});raise
