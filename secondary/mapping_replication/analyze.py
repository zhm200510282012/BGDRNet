"""All-three-seed descriptive paired analysis; no p-values, plots, or seed selection."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json
import numpy as np
R=Path(__file__).resolve().parent
KEYS=['mIoU','glass_IoU','boundary_F1','non_glass_FPR','glass_FNR','MAE']
SEEDS=[42,43,44]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,o):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def stats(values):
    return {'n':len(values),'mean':float(np.mean(values)),'sample_sd_ddof1':float(np.std(values,ddof=1)) if len(values)>1 else None,'values':values}
def fm(v):return f'{v:.3f}'
def main():
    f=read(R/'execution.frozen.json');status=read(R/'status.json');records={};epochs={};histories={}
    entries={**f['references'],**f['jobs']}
    for key,j in entries.items():
        n=j['variant'];p=Path(j['path'])/f'runs/{n}/result.json'
        if not p.exists():continue
        result=read(p);records[key]={'development':result['validation']};epochs[key]=result['best_epoch']
        h=[json.loads(x) for x in (p.parent/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if x]
        assert len(h)==40 and all(x['global_step']==271*x['epoch'] for x in h)
        histories[key]={'epoch40':h[-1]['validation'],'last5_mIoU':[x['validation']['mIoU'] for x in h[-5:]],'best_epoch':epochs[key]}
        for d in f['evaluation_order']:
            root=Path(j['evaluation_source_root'])/d/n if key in f['references'] else R/'evaluation'/d/key
            rp=root/'result.json'
            if rp.exists():
                out=read(rp);assert out['checkpoint_epoch']==epochs[key] and out['threshold']==.5
                records[key][d]=out['metrics']
    summary={'status':status['status'],'seeds':SEEDS,'primary_unchanged':'FR29_seed42','per_seed':records,'best_epochs':epochs,
             'histories':histories,'aggregate':{},'paired':{},'plots':False,'test_tuning':False}
    lines=['# FR28 / FR29 最小种子稳定性报告','',f"状态：**{status['status']}**。更新时间UTC：{datetime.now(timezone.utc).isoformat()}。",'',
        '固定两模型、40轮、seed42/43/44，42复用，新增43/44各两模型。训练与数据/标签/阈值不变，预先指定FR29_seed42为主模型，不按复验测试分数改选。不新增模块、种子、阈值或绘图。',
        '', '## 原始三种子结果与汇总','']
    csvrows=[]
    for dataset in ['development',*f['evaluation_order']]:
        lines.extend([f'### {dataset}','','| 模型 | seed | best epoch | mIoU% | glassIoU% | BF1% | FPR%↓ | FNR%↓ | MAE↓ |','|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
        for n in ['FR28','FR29']:
            values=[]
            for seed in SEEDS:
                key=f'{n}_seed{seed}'
                if dataset not in records.get(key,{}):continue
                m=records[key][dataset];values.append(m)
                lines.append(f'| {n} | {seed} | {epochs[key]} | '+' | '.join([fm(100*m[k]) for k in KEYS[:-1]]+[f"{m['MAE']:.5f}"])+' |')
                csvrows.append({'dataset':dataset,'variant':n,'seed':seed,'best_epoch':epochs[key],**{k:m[k] for k in KEYS}})
            summary['aggregate'].setdefault(dataset,{})[n]={k:stats([m[k] for m in values]) for k in KEYS} if values else {}
            if len(values)==3:
                arr=summary['aggregate'][dataset][n]
                lines.append(f'| {n} mean±SD | 3seeds | — | '+' | '.join([f"{100*arr[k]['mean']:.3f}±{100*arr[k]['sample_sd_ddof1']:.3f}" for k in KEYS[:-1]]+[f"{arr['MAE']['mean']:.5f}±{arr['MAE']['sample_sd_ddof1']:.5f}"])+' |')
        paired={};paired_seeds=[]
        for seed in SEEDS:
            a=records.get(f'FR28_seed{seed}',{}).get(dataset);b=records.get(f'FR29_seed{seed}',{}).get(dataset)
            if a is not None and b is not None:
                paired_seeds.append(seed)
                for k in KEYS:paired.setdefault(k,[]).append(b[k]-a[k])
        summary['paired'][dataset]={k:{**stats(v),'seeds':paired_seeds,'positive_count':sum(x>0 for x in v),'negative_count':sum(x<0 for x in v)} for k,v in paired.items()}
        lines.extend(['','FR29−FR28配对差值；mIoU/IoU/BF1正数有利，FPR/FNR/MAE负数有利。','',
                      '| 指标 | seed42 | seed43 | seed44 | mean±sample SD |','|---|---:|---:|---:|---:|'])
        for k in KEYS:
            p=summary['paired'][dataset].get(k)
            if not p:continue
            scale=1 if k=='MAE' else 100;lookup=dict(zip(p['seeds'],p['values']))
            meansd=f"{scale*p['mean']:+.4f}±{scale*p['sample_sd_ddof1']:.4f}" if p['n']==3 else 'PENDING'
            lines.append('| '+k+(' raw' if k=='MAE' else ' pp')+' | '+' | '.join(f'{scale*lookup[s]:+.4f}' if s in lookup else 'NOT_RUN' for s in SEEDS)+' | '+meansd+' |')
    all_complete=all(summary['paired'].get(d,{}).get('mIoU',{}).get('n')==3 for d in ['development',*f['evaluation_order']])
    summary['all_three_seed_endpoints_complete']=all_complete
    lines.extend(['','## 稳定性解释',''])
    if all_complete:
        for d in ['development',*f['evaluation_order']]:
            p=summary['paired'][d]['mIoU'];bf=summary['paired'][d]['boundary_F1'];fn=summary['paired'][d]['glass_FNR'];fp=summary['paired'][d]['non_glass_FPR']
            lines.append(f"- {d}：mIoU均值差{100*p['mean']:+.3f}pp（正向{p['positive_count']}/3）；BF1均值差{100*bf['mean']:+.3f}pp（正向{bf['positive_count']}/3）；FNR均值差{100*fn['mean']:+.3f}pp、FPR均值差{100*fp['mean']:+.3f}pp。")
        target=summary['paired']['GEGD2']
        if target['boundary_F1']['mean']<=0:lines.append('GEGD2平均边界净收益没有成立，不写稳定跨域边界改进；正向源域指标不能替代此端点。')
        if any(summary['paired'][d]['mIoU']['positive_count']<3 for d in ['development','GSD_test']):lines.append('源域mIoU存在种子方向不一致，独立输出的普适改进主张需要收窄；完整保留三个种子，不筛选或追加。')
        lines.append('n=3仅描述本训练随机性范围；没有给训练层面显著性或接收概率保证。原2×2只有seed42，本轮只复验差分条件FR28/29，不能冒称完整2×2交互也已做三种子。')
    else:lines.append('结果尚未齐全，暂不判断种子稳定性；未完成项保持NOT_RUN。')
    lines.extend(['','## 公平性、历史暴露与资源','',
        '所有模型同训练划分/预训练/监督/40轮；两模型差别为首级共享或独立输出映射，后者多1024参数。同seed公共初始化/RNG与数据流配对。旧seed42来自先前会话，不能消除历史开发选择偏差。',
        '', 'GSD test813及GEGD2_950均已有历史观察，本次为固定结构的训练随机性复核，不是首次盲测；不以结果修改结构/超参/最佳轮规则。原尺寸GT/后处理固定，GSD raw>0、GEGD2 raw==255，extra579不使用。',
        '', '每模型均值±样本标准差ddof1反映三个种子，不把epoch/像素/图像当独立训练重复，不用既有图像bootstrap替代种子方差。参数与资源沿用既有同会话结构对照，不跨会话挑最快值；新job原始profile保留但不当推广门槛。',
        '', '## 交付与下一步','',
        'execution.frozen.json / PROTOCOL.md记录配置、源码、顺序和清单；checkpoints.frozen.json在四组新fit全部完成后、首次新终点预测前冻结。best/last/deployment及逐epoch日志/curves.csv位于D:/大创/cache/glass_seedrep_B40_20261029/<job>/runs/<variant>/。evaluation/<dataset>/<job>/逐图CSV按100图恢复；seed42直接引用旧结果。',
        '', '四fit与两库评估完成即停止新训练。按全部三种子结果决定稿件主张范围，保留原始效用门控和跨域负结果；不添加FR33、更多种子或科研图表。复算执行run.ps1 -Mode all，拒绝源/配置/清单混用。'])
    (R/'FEASIBILITY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save(R/'summary.json',summary)
    with (R/'per_seed_metrics.csv').open('w',newline='',encoding='utf-8-sig') as fcsv:
        w=csv.DictWriter(fcsv,fieldnames=['dataset','variant','seed','best_epoch',*KEYS]);w.writeheader();w.writerows(csvrows)
    print('Updated '+str(R/'FEASIBILITY_REPORT.md'))
if __name__=='__main__':main()
