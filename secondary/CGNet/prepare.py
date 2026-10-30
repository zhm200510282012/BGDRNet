"""Prepare once; never overwrite a pre-existing experiment or reference artifact."""
from pathlib import Path
import json, shutil, subprocess
from datetime import datetime, timezone
R=Path(__file__).resolve().parent
REF=Path('D:/大创/cache/glass_B40_FR29_20261028/FR29')
PREFIX=Path('D:/SIVP/glass_s2_operatingpoint_20261025/pretrained_decoder_control')
CACHE=Path('D:/大创/cache/glass_external_B40_20261028')

def save(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    assert not (R/'config.frozen.json').exists(), 'Already prepared; use run.ps1 to resume'
    assert not CACHE.exists(), 'Choose a new cache; existing files are protected'
    CACHE.mkdir(parents=True)
    (R/'logs').mkdir(exist_ok=True)
    for filename in ['runtime.py','manifest.csv']:
        shutil.copyfile(REF/filename,R/filename)
    shutil.copyfile(PREFIX/'s2_frozen.py', R/'prefix_s2_frozen.py')
    source=(PREFIX/'model.py').read_text(encoding='utf-8')
    (R/'prefix_model.py').write_text(source.replace('from s2_frozen import','from prefix_s2_frozen import'),encoding='utf-8')
    reference=R/'reference';reference.mkdir()
    refnames=['model.py','fr12_frozen.py','fr11_frozen.py','flow_base_frozen.py','fr2_frozen.py','fna_frozen.py','pilot_frozen.py','s2_frozen.py']
    for name in refnames:shutil.copyfile(REF/name,reference/name)
    cfg=json.loads((REF/'config.frozen.json').read_text(encoding='utf-8'))
    shared={k:cfg[k] for k in ['size','epochs','seed','split_seed','batch_size','lr','weight_decay','threads','workers','augmentation','checkpoint_selection']}
    shared.update({'created_utc':datetime.now(timezone.utc).isoformat(),'cache':str(CACHE),'reference_path':str(REF),
        'weights':cfg['weights'],'candidate_order':['CGNet','HARD-XS','EFRNet_ICME2021','LRASPP_prefix'],
        'runnable':['CGNet','LRASPP_prefix'],'scheduler':'CosineAnnealingLR T_max40 eta_min0, step after epoch',
        'optimizer':'AdamW, betas .9/.999 eps1e-8, global norm clip5',
        'precision':'AMP BF16 with FP32 weights; no GradScaler; TF32 disabled; strict deterministic',
        'loss':'BCEWithLogits + per-image soft IoU; no invented auxiliary heads',
        'threshold':.5,'official_test':'NO_ACCESS_NO_EVALUATION',
        'initialization':{'CGNet':'Official kaiming initialization; fresh scratch; no segmentation warm start',
            'LRASPP_prefix':'Fresh cached ImageNet MobileNetV2 prefix + random unmodified torchvision LRASPPHead'},
        'batchnorm':{'CGNet':'Native BN running statistics and affine parameters trained',
            'LRASPP_prefix':'Encoder running statistics frozen, affine trained; decoder BN trained'},
        'parameter_limit':'No pruning or narrowing of official candidates; full binary inference model counted; same order of magnitude',
        'benchmark':{'size':256,'batch':1,'precision':'AMP BF16, FP32 weights','warmup':60,'rounds':4,'samples_per_round':40,
            'orders':'forward/reverse/reverse/forward; alternate models after each sample',
            'timing':'synchronized host perf_counter; model-only with normalization/output resize, excludes disk/decode/native postprocess',
            'compute':'2*(Conv2d+Linear+attention QK/AV MACs); excludes resize/norm/elementwise/pooling',
            'memory':'Fresh subprocess per model, absolute allocated/reserved peaks and incremental activation memory; benchmark shared-residency memory reported separately'},
        'resume':'Full epoch boundaries; model+optimizer+scheduler+RNG+source+config+manifest checked',
        'plotting':False,'replications':1,'fixed_budget_limitation':'40 epochs is not a convergence ceiling or original-paper reproduction'})
    save(R/'config.frozen.json',shared)
    history=[json.loads(x) for x in (REF/'runs/FR29/epochs.jsonl').read_text(encoding='utf-8').splitlines() if x]
    assert [x['epoch'] for x in history]==list(range(1,41))
    assert all(x['global_step']==x['epoch']*271 for x in history)
    result=json.loads((REF/'runs/FR29/result.json').read_text(encoding='utf-8'))
    best=max(history,key=lambda x:(x['validation']['mIoU'],-x['epoch']))
    assert best['epoch']==result['best_epoch'] and best['validation']==result['validation']
    assert result['efficiency']['parameters']==278804
    save(R/'reference_reuse.json',{'status':'CHECKED_HISTORY_AND_METRICS','checkpoint_loading':'DEFERRED_TO_SMOKE',
        'reference':str(REF),'result':result,'history':history,'same_manifest':True,
        'training_parameters':278903,'inference_parameters':278804,'retraining':False,
        'pretraining':'ImageNet MobileNetV2 prefix features[:11]',
        'auxiliary_supervision':'FR29 retained regional and boundary heads, weights .2/.2; removed where unused at inference',
        'important':'Only practical fixed-budget system comparison; no isolated architecture/pretraining/auxiliary-loss attribution'})
    env={'created_utc':shared['created_utc'],'cwd':str(R),'python':__import__('sys').executable,
         'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,
         'disk_free_GiB':shutil.disk_usage('D:/').free/2**30,'previous_external_queue':'STOPPED by user; untouched',
         'current_main_task':'FR32 COMPLETE, FR29 external comparison newly authorized; no existing matching 40-epoch CGNet/prefix run',
         'data_root':'D:/DSP/GSD','no_data_download':True}
    save(R/'audit/environment.json',env)
    # Save exact sources instead of repeated hashing; these must match on every resume.
    names=['models.py','prefix_model.py','prefix_s2_frozen.py','runtime.py','runner.py','controller.py','analyze.py']
    sources={name:(R/name).read_text(encoding='utf-8') for name in names}
    sources.update({str(p.relative_to(R)):p.read_text(encoding='utf-8') for p in list(reference.glob('*.py'))+[R/'vendor/CGNet.py']})
    save(R/'execution.frozen.json',{'config':shared,'manifest':(R/'manifest.csv').read_text(encoding='utf-8'),
        'sources':sources,'protocol':(R/'PROTOCOL.md').read_text(encoding='utf-8')})
    save(R/'status.json',{'status':'PREPARED_NOT_SMOKED','formal_training_started':False})
    print(json.dumps(env,ensure_ascii=False))

if __name__=='__main__':main()
