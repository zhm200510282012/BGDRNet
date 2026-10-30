"""Real-data smoke and a sequential single-seed screen; no test access or plots."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import TinyGlass, VARIANTS, count_parameters, auxiliary_targets, boundary_band, loss_bundle
import runtime

BASE=Path(__file__).resolve().parent
CFG=json.loads((BASE/'config.frozen.json').read_text(encoding='utf-8'))
MANIFEST=(BASE/'manifest.csv').read_text(encoding='utf-8')
SOURCES={name:(BASE/name).read_text(encoding='utf-8') for name in ['model.py', 'runtime.py', 'fr12_frozen.py', 'fr11_frozen.py', 'flow_base_frozen.py', 'fr2_frozen.py', 'fna_frozen.py', 'pilot_frozen.py', 's2_frozen.py', 'experiment.py', 'worker.py']}
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


def smoke(data):
    result={'timestamp':runtime.stamp(),'status':'RUNNING','data':'real GSD training images and glass masks',
            'config':CFG,'arms':{},'test_access':False,'purpose':'engineering only; no generalization claims'}
    x,y=data.batch(data.splits['train'][:CFG['batch_size']])
    assert x.shape==(8,3,256,256) and set(torch.unique(y).tolist())<={0.,1.}
    common=None; routed_initial=None
    for variant in VARIANTS:
        runtime.seed_all(CFG['seed'])
        model=TinyGlass(variant,CFG['weights'],training_heads=True).to(DEVICE)
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith(('edge_heads.','body_heads.'))}
        if common is None: common=before
        else: assert common.keys()==before.keys() and all(torch.equal(common[k],before[k]) for k in common)
        if variant=='S2': routed_initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if variant=='S3': assert all(torch.equal(v.detach().cpu(),routed_initial[k]) for k,v in model.state_dict().items())
        model.eval()
        # Verify true ImageNet tensor transfer, not merely the presence of a cached filename.
        weights=torch.load(CFG['weights'],map_location='cpu',weights_only=True)
        assert all(torch.equal(v.detach().cpu(),weights['features.'+k]) for k,v in model.encoder.state_dict().items())
        del weights,before
        if model.options['route']:
            model.train(); model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(x,with_aux=True)
                seg=F.binary_cross_entropy_with_logits(prediction['logits'],y)
            seg.backward()
            assert all(p.grad is None for head in model.edge_heads for p in head.parameters()), 'Segmentation gradient leaked through detached gate'
            del prediction,seg
        optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG['epochs'])
        steps=[]
        for _ in range(3):
            start=time.monotonic(); loss,norm,meta=checked_update(model,optimizer,x,y); torch.cuda.synchronize()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            if model.options['complementary']:
                assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()) for head in model.edge_heads)
                assert all(any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()) for head in model.body_heads)
                if model.options['soft']: assert meta['soft_fraction_count']>0
                else: assert meta['soft_fraction_count']==0
            steps.append({'loss':loss,'gradient_norm':norm,'seconds':time.monotonic()-start,**meta})
        out=BASE/'smoke'/variant; out.mkdir(parents=True,exist_ok=True)
        save_checkpoint(out/'last.pt',make_checkpoint(model,optimizer,scheduler,0,-1.))
        saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False); verify_checkpoint(saved,variant)
        model2=TinyGlass(variant,training_heads=True).to(DEVICE); model2.load_state_dict(saved['model'])
        optimizer2=torch.optim.AdamW(model2.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']); optimizer2.load_state_dict(saved['optimizer'])
        for m,o in [(model,optimizer),(model2,optimizer2)]:
            runtime.restore_rng(saved['rng']); checked_update(m,o,x,y)
        diff=max(float((a-b).abs().max()) for a,b in zip(model.parameters(),model2.parameters()))
        assert diff<=1e-6,diff
        model.eval()
        deployed=model.deployment_copy().to(DEVICE).eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            deploy_diff=float((model(x)-deployed(x)).abs().max())
        assert deploy_diff==0
        deployed_params=count_parameters(deployed)
        assert deployed_params<=CFG['inference_parameter_ceiling']
        save_checkpoint(out/'deployment.pt',{'model':deployed.cpu().state_dict(),'variant':variant,'config':CFG,'smoke_only':True})
        result['arms'][variant]={'training_parameters':count_parameters(model),'inference_parameters':deployed_params,
            'finite_loss_and_gradients':True,'same_common_initialization':True,'pretrained_tensors_verified':True,
            'gate_segmentation_gradient_blocked':model.options['route'],'checkpoint_resume_next_update_max_abs_diff':diff,
            'auxiliary_head_pruning_max_abs_diff':deploy_diff,'steps':steps}
        save_json(BASE/'logs/smoke_partial.json',result)
        del model,model2,optimizer,optimizer2,scheduler,deployed,saved
        torch.cuda.empty_cache()
    # The image perimeter is ignored rather than inventing a contour outside the image.
    assert boundary_band(torch.ones(1,1,16,16,device=DEVICE),2).sum()==0
    result['status']='PASS'; result['source_text']=SOURCES
    save_json(BASE/'logs/smoke.json',result)
    print(json.dumps({'event':'smoke_complete','status':'PASS','arms':result['arms']}),flush=True)


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


def baseline_gate(result):
    v=result['validation']; e=result['efficiency']; floors=CFG['baseline_gate']
    checks={'mIoU':v['mIoU']>=floors['min_validation_mIoU'],
            'glass_IoU':v['glass_IoU']>=floors['min_validation_glass_IoU'],
            'parameters':e['parameters']<=CFG['inference_parameter_ceiling']}
    return {'pass':all(checks.values()),'checks':checks,'floors':floors,'split':'development validation only'}


def verify_results(results,gate):
    expected=['S0','S1','S2','S3'] if gate['pass'] else ['S0']
    assert [r['variant'] for r in results]==expected
    for r in results:
        out=BASE/'runs'/r['variant']
        records=[json.loads(line) for line in (out/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if line]
        epochs={x['epoch']:x for x in records}
        assert sorted(epochs)==list(range(1,CFG['epochs']+1))
        best=max(epochs.values(),key=lambda x:(x['validation']['mIoU'],-x['epoch']))
        assert best['epoch']==r['best_epoch'] and best['validation']==r['validation']
        assert r['validation']['images']==542
        last=torch.load(out/'last.pt',map_location='cpu',weights_only=False); verify_checkpoint(last,r['variant'])
        assert last['epoch']==CFG['epochs']; del last
        assert CFG.get('inference_parameter_ceiling') is None or r['efficiency']['parameters']<=CFG['inference_parameter_ceiling']
        deployed=torch.load(out/'deployment.pt',map_location='cpu',weights_only=False)
        inference=TinyGlass(r['variant'],training_heads=False)
        inference.load_state_dict(deployed['model'],strict=True)
        assert count_parameters(inference)==r['efficiency']['parameters']==deployed['parameters']
        del deployed,inference
    if gate['pass']:
        assert results[0]['efficiency']['parameters']==results[1]['efficiency']['parameters']
        assert results[2]['efficiency']['parameters']==results[3]['efficiency']['parameters']
    return {'status':'VERIFIED','variants':expected,'epochs_each':CFG['epochs'],'test_evaluation':False}


def write_report(status,results=None,gate=None,error=None):
    results=results or []
    lines=['# 0.30M以内轻量玻璃分割：最小可行性验证','',f'更新时间：{runtime.stamp()}；状态：**{status}**。',
      '', '## 范围与数据',
      '本轮仅做真实数据smoke和seed42开发筛选。使用已审计GSD官方train中的2168训练/542开发验证，保持旧清单成员；新manifest完全排除官方test。没有读取本轮测试图像或标签，没有使用extra、GEGD2或下载新资源，也没有绘制科研图表。',
      '该验证集已在先前开发中使用，结论属于开发筛选；没有场景ID，不能保证同场景独立。上一轮效用门控D未过门槛的结果原样保留，本轮为新的结构假设。',
      '', '## 固定对照',
      '| 组 | 互补监督 | 边界差分路由 | 边界目标 |',
      '|---|---|---|---|',
      '| S0 | 无 | 无；R+DSConv(L) | 无 |',
      '| S1 | 区域+边界 | 无；同S0主路径 | 各级硬边界 |',
      '| S2 | 同S1 | R+sg(E)*DSConv(L-UH) | 同S1硬边界 |',
      '| S3 | 同S1 | 与S2完全相同 | 原训练画布边界带面积缩放的软标签 |',
      'S1的区域头和边界头仅用于训练；部署时均剥离。S2/S3保留边界头用于门控，仅剥离区域辅助头。所有实际推理参数都计入上限，剥离前后用真实样本检查输出一致。',
      'S1→S2同时加入差分和路由，这是一个模块级对照，不能分别认定作差和门控各自有效；S2→S3同时涉及尺度一致定义和硬→软目标，不能用四组独立识别两者贡献。',
      '', '## 配置与公平性',
      '全部组：同一ImageNet MobileNetV2前11模块、相同公共张量初始化、相同训练/验证成员、256x256、20epochs、batch8、AdamW lr3e-4/wd0.01、cosine、BF16、梯度裁剪5、nearest上采样、严格确定性；主干BN统计冻结而affine训练，解码器GroupNorm。',
      'RGB双线性缩放、mask最近邻；水平翻转p=0.5；每epoch独立固定的数据顺序和增强随机流在全部组相同。没有从上一组检查点继续训练新组，全部从同一初始化重新开始。',
      '最终损失全组固定BCE+softIoU；S1–S3另加0.2*区域互补损失+0.2*边界BCE/Dice。区域标签全组同用面积缩放，区域权重为1-B；区别只在所列模块和边界目标。',
      '旧大基线A的decoder和BCE-only损失不同，只作为“是否值得继续”的工程锚点，不能把新旧差异全部归因于截断主干。',
      '', '## Smoke与断点',
      'smoke使用真实训练图和mask；检查全部组的有限损失/梯度、预训练张量身份、公共初始化、硬/软标签、边界头分割梯度截断、辅助头有效梯度、保存/恢复后的同一下一步更新与部署剥离前后一致性。smoke成绩不作为泛化结果。',
      'logs/smoke.json为通过依据；runs/S*/last.pt含模型、optimizer、scheduler、epoch、RNG、完整配置/清单及精确源码文本。仅完整epoch边界恢复；途中打断重做当epoch，拒绝混配置/混源码恢复。每epoch仅写last、最佳改善时写best、并写一条JSONL。',
      '', '## 基线门槛',
      f'正式新结果产生前确定：S0验证mIoU≥{CFG["baseline_gate"]["min_validation_mIoU"]:.6f}、玻璃IoU≥{CFG["baseline_gate"]["min_validation_glass_IoU"]:.6f}、推理参数≤300000。',
      '这对应旧A验证mIoU最多下降2个百分点、玻璃IoU最多下降3个百分点。门槛是工程判断，不是显著性检验或发表标准。S0不通过则停止添加模块；通过后按相同预算完成S1–S3，不根据中间组成绩调整配置。本轮不扩展多种子。',
      '', '## 开发验证结果']
    if not results:
        lines+=['尚无正式验证指标；不得以smoke训练损失或静态参数替代。']
    else:
        lines+=['所有指标来自最佳验证mIoU检查点，精确并列保留较早epoch；不属于测试集结果。',
            '| 组 | 最佳epoch | mIoU | 玻璃IoU | MAE | BoundaryF1 | 非玻璃FPR | 玻璃FNR | 推理参数 | 训练参数 |',
            '|---|---|---|---|---|---|---|---|---|---|']
        for r in results:
            v=r['validation']; e=r['efficiency']
            lines.append(f'| {r["variant"]} | {r["best_epoch"]} | {v["mIoU"]:.6f} | {v["glass_IoU"]:.6f} | {v["MAE"]:.6f} | {v["boundary_F1"]:.6f} | {v["non_glass_FPR"]:.6f} | {v["glass_FNR"]:.6f} | {e["parameters"]} | {r["training_parameters"]} |')
        lines+=['','| 组 | 卷积FLOPs/G | 延迟中位ms | p95ms | 推理基线/峰值MiB | 前向增量MiB | 训练+验证峰值MiB |',
                '|---|---|---|---|---|---|---|']
        for r in results:
            e=r['efficiency']
            lines.append(f'| {r["variant"]} | {e["conv_FLOPs_2MACs"]/1e9:.4f} | {e["latency_ms_median"]:.3f} | {e["latency_ms_p95"]:.3f} | {e["inference_baseline_allocated_MiB"]:.2f}/{e["inference_peak_allocated_MiB"]:.2f} | {e["inference_incremental_allocated_MiB"]:.2f} | {r["train_val_peak_allocated_MiB"]:.2f} |')
        if len(results)>1:
            lines+=['','相邻组mIoU变化（百分点）：'+', '.join(f'{b["variant"]}−{a["variant"]}: {100*(b["validation"]["mIoU"]-a["validation"]["mIoU"]):+.3f}' for a,b in zip(results,results[1:]))+'。']
    lines+=['','## 指标口径与资源测量',
      'logits恢复原图尺寸后sigmoid、固定阈值0.5；全验证像素累计confusion计算玻璃/背景IoU均值；MAE为概率像素平均绝对误差，FPR=FP/(FP+TN)，FNR=FN/(TP+FN)。',
      'Boundary F1沿用上一轮实现：3x3腐蚀提取内轮廓，忽略图外边界；欧氏匹配容差max(1,ceil(0.0075*原图对角线))，逐图宏平均，双空=1、单空=0。',
      '每组在独立新子进程测部署模型，先释放训练态；同一RTX4070Laptop/BF16/batch1，50次预热+200次CUDA事件同步计时；不包含图像读取、CPU解码及指标计算。峰值allocated、基线和前向增量分开列出，reserved保存在JSON。',
      'FLOPs为卷积MACx2，不含归一化、bias、池化、激活、上采样和逐元素操作，不能称完整算子FLOPs。',
      '', '## 继续/停止结论']
    if gate is not None:
        lines.append('S0门槛：`'+json.dumps(gate,ensure_ascii=False)+'`。')
        if not gate['pass']:
            lines.append('建议：STOP_BASELINE。当前轻量基线未达到事先定义的精度底线；S1–S3记NOT_RUN，不把未跑模块写为失败。先报告截断后整套基线的容量/优化问题，不在本轮擅自增加主干、改轮数或用测试集挑方案。')
        elif len(results)==4:
            best=max(results,key=lambda r:r['validation']['mIoU'])
            lines.append(f'单种子最佳开发mIoU组为{best["variant"]}。每项改动按实际差值单独解释，不能因为最终模型或个别指标改善便认定所有模块有效。')
            s0,s1,s2,s3=results
            gain=s3['validation']['mIoU']-s0['validation']['mIoU']
            b_gain=s3['validation']['boundary_F1']-s0['validation']['boundary_F1']
            lines.append('最终相对S0：mIoU变化'+f'{100*gain:+.3f}个百分点，BoundaryF1变化{100*b_gain:+.3f}个百分点。')
            lines.append('这仍是开发筛选结果；是否进行多种子/独立数据验证需要下一阶段明确范围。没有以已见GSD测试集补充调参或声称新泛化结论。')
    else: lines.append('等待真实smoke/基线结果；目前不作方向有效性判断。')
    if error: lines+=['','## 失败/阻塞','```text',error,'```']
    lines+=['','## 产物与复现',
      '冻结配置config.frozen.json；训练/验证清单manifest.csv；smoke在logs/smoke.json与smoke/；正式日志runs/S*/epochs.jsonl；last.pt用于恢复，best.pt用于最佳开发模型；deployment.pt为去除训练辅助头后的模型；results.json、baseline_gate.json、logs/final_verification.json记录聚合与核验。',
      '恢复时执行run.ps1，它会拒绝重复运行进程；不改冻结配置或代码继续混跑。原工程、设计预算原型和上一轮报告保持原位。',
      '本流程采用experimental-design技能的固定对照与重复层级约束。软件/流程参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)。该参考不是玻璃方法有效性的证据。']
    (BASE/'FEASIBILITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['smoke','run','profile'],required=True)
    parser.add_argument('--variant',choices=list(VARIANTS)); args=parser.parse_args()
    torch.set_num_threads(CFG['threads']); cv2.setNumThreads(2)
    if args.mode=='profile':
        assert args.variant; profile(args.variant); return
    results=[]; gate=None
    try:
        data=data_cache()
        if args.mode=='smoke':
            smoke(data); write_report('SMOKE_PASS_FORMAL_NOT_STARTED'); return
        passed=json.loads((BASE/'logs/smoke.json').read_text(encoding='utf-8'))
        assert passed['status']=='PASS' and passed['config']==CFG and passed['source_text']==SOURCES
        write_report('BASELINE_RUNNING')
        result=train_one('S0',data); results.append(result); save_json(BASE/'results.json',results)
        gate=baseline_gate(result); save_json(BASE/'baseline_gate.json',gate)
        write_report('BASELINE_PASS_ABLATIONS_PENDING' if gate['pass'] else 'COMPLETE_STOP_BASELINE',results,gate)
        print(json.dumps({'event':'baseline_gate','gate':gate}),flush=True)
        if gate['pass']:
            for variant in ['S1','S2','S3']:
                write_report('ABLATIONS_RUNNING',results,gate)
                results.append(train_one(variant,data)); save_json(BASE/'results.json',results)
                write_report('ABLATIONS_RUNNING',results,gate)
        verification=verify_results(results,gate); save_json(BASE/'logs/final_verification.json',verification)
        status='COMPLETE_ALL_STAGES' if gate['pass'] else 'COMPLETE_STOP_BASELINE'
        write_report(status,results,gate)
        save_json(BASE/'logs/completion.json',{'timestamp':runtime.stamp(),'status':status,'baseline_gate_pass':gate['pass'],'variant_count':len(results),'test_access':False})
        print(json.dumps({'event':'complete','status':status,'report':str(BASE/'FEASIBILITY_REPORT.md')}),flush=True)
    except Exception:
        error=traceback.format_exc(); (BASE/'logs/failure.txt').write_text(error,encoding='utf-8')
        write_report('FAILED_REQUIRES_REVIEW',results,gate,error); raise


if __name__=='__main__': main()
