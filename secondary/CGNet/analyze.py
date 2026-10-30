"""Render numeric tables and bounded interpretation; never generate plots."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json
R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def fmt(x,d=3):return f'{x:.{d}f}' if isinstance(x,(int,float)) else 'NOT_RUN'

def main():
    cfg=read(R/'config.frozen.json');cache=Path(cfg['cache']);audit=read(R/'source_audit.json')
    reuse=read(R/'reference_reuse.json');ref=reuse['result'];state=read(R/'status.json')
    latency=read(R/'matched_latency.json') if (R/'matched_latency.json').exists() else {'results':{}}
    models={'FR29':{'model':'FR29','status':'REUSED_COMPLETE','best_epoch':ref['best_epoch'],'validation':ref['validation'],
        'training_parameters':278903,'inference_parameters':278804,'history':reuse['history']}}
    for n in cfg['runnable']:
        p=cache/n/'result.json'
        if p.exists():models[n]=read(p)
    for n,m in models.items():
        p=cache/n/'efficiency.json';m['efficiency']=read(p) if p.exists() else None
        m['matched_latency']=latency['results'].get(n)
    fields=['model','status','best_epoch','mIoU','glass_IoU','MAE','boundary_F1','non_glass_FPR','glass_FNR',
        'inference_parameters','training_parameters','matrix_FLOPs_2MACs','latency_median_ms','latency_p95_ms',
        'inference_peak_allocated_MiB','inference_peak_reserved_MiB','activation_increment_MiB']
    rows=[]
    for n in ['FR29',*cfg['candidate_order']]:
        if n not in models:
            rows.append({'model':n,'status':audit['models'][n]['status'] if n in audit['models'] else 'NOT_RUN'});continue
        m=models[n];v=m['validation'];e=m.get('efficiency') or {};l=m.get('matched_latency') or {}
        rows.append({'model':n,'status':m['status'],'best_epoch':m['best_epoch'],**{k:v[k] for k in fields if k in v},
            'inference_parameters':m['inference_parameters'],'training_parameters':m['training_parameters'],
            'matrix_FLOPs_2MACs':e.get('matrix_FLOPs_2MACs'),'latency_median_ms':l.get('median_ms'),'latency_p95_ms':l.get('p95_ms'),
            'inference_peak_allocated_MiB':e.get('peak_allocated_MiB'),'inference_peak_reserved_MiB':e.get('peak_reserved_MiB'),
            'activation_increment_MiB':e.get('incremental_activation_MiB')})
    with (R/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    full=state['status'] in ['COMPLETE','COMPLETE_WITH_FAILURES'] and bool(latency['results'])
    text=['# FR29 同量级轻量模型可行性报告','',f"状态：**{state['status']}**。更新时间UTC：{datetime.now(timezone.utc).isoformat()}。官方测试 NOT_RUN；无科研图表。",'',
        '## 任务与实现审计','',
        'FR29 使用已完成的 seed42/40轮最佳checkpoint，不重复训练；旧 external_classic_baselines_20261026 队列保持用户停止状态。新实验与缓存均独立。主线程FR32已完成，启动前未发现其他Python训练进程，RTX4070 Laptop约6.8GiB显存可用；具体快照见 audit/environment.json。',
        '','| 模型 | 实现身份 / 状态 | 预训练 |','|---|---|---|',
        '| FR29 | 现有冻结实现；278804推理参数；复用epoch28 | 已缓存ImageNet MobileNetV2 prefix |',
        '| CGNet | [作者MIT代码](https://github.com/wutianyiRosun/CGNet)，commit cfccea377a6f0a659e807e8759c82dbffba02d7f，M3N21、classes2、dropout False；原文件无结构修改 | 作者scratch初始化；README提供Baidu分割权重链接，本轮未下载或使用；权重文件内容未验证 |',
        '| HARD-XS | **BLOCKED**：ACCV2024论文身份已核实；在论文页面、作者实验室GitHub组织与公开仓库检索中未取得可确认的作者实现、XS配置及权重 | UNKNOWN；未下载 |',
        '| EFRNet_ICME2021 | **BLOCKED**：Kuayue Zhang等，ICME2021 Efficient Feature Fusion and Refinement；未取得作者可靠代码。zhuyr97/EFRNet是NTIRE2023超分辨率，已排除 | UNKNOWN；未下载 |',
        '| LRASPP_prefix | 复用旧适配实现，完整2类LRASPPHead + MobileNetV2 features[:11]，不是完整官方LR-ASPP | 原始ImageNet初始化重新训练40轮；不复用旧20轮任务权重 |',
        '', '来源：[HARD官方论文页](https://openaccess.thecvf.com/content/ACCV2024/html/Kwon_HARD__Hardware-Aware_lightweight_Real-time_semantic_segmentation_model_Deployable_from_ACCV_2024_paper.html)、[作者实验室组织](https://github.com/EmPasLab)、[指定EFRNet作者机构全文](https://discovery.ucl.ac.uk/id/eprint/10129651/1/KuayueZhang-ICME2021-final.pdf)。BLOCKED表示本次有界检索未取得可靠实现，不能据此断言作者从未公开代码。',
        '','## 冻结配置与公平性','',
        '共同条件：GSD官方训练集拆分2168/542、split_seed20261025、size256、seed42、40epoch、batch8、每轮271更新、AdamW lr3e-4/wd.01、cosine40、BCE+每图softIoU、水平翻转.5、BF16/FP32权重、严格确定性、相同原尺寸评估。最大验证global mIoU选best、精确并列选早轮。详见 PROTOCOL.md/config.frozen.json。',
        '',
        '差异必须保留：FR29与prefix有ImageNet预训练，CGNet为scratch；FR29有区域/边界辅助监督，其他两者保留原结构不新增辅助头；CGNet BN统计可训，两个prefix的encoder BN统计冻结；上采样保留各自设计；CGNet原作者使用更长预算、不同优化器/类别损失/随机缩放。本表只能描述共同40轮训练配方下的系统取舍，不能证明纯架构胜负或原方法性能上限。',
        '','## 真实数据 smoke','']
    if (R/'smoke_summary.json').exists():
        s=read(R/'smoke_summary.json');text.append(f"状态 **{s['status']}**。8张真实训练图，有限损失/梯度、全部参数参与、输出尺寸、预训练转移、模型/optimizer/scheduler/RNG恢复及部署输出等价；另16张固定验证图只检查指标引擎。正式训练重新初始化。")
        for n,a in s['arms'].items():text.append(f"- {n}：完整推理{a['inference_parameters']}参数，恢复下一更新最大差{a['checkpoint_next_update_max_abs_diff']:.3g}，部署最大差{a['deployment_output_max_abs_diff']:.3g}。")
        text.append(s['reference_checkpoint'])
    else:text.append('NOT_RUN。正式训练尚未启动。')
    text.extend(['','## 验证集质量（全部来自各自同一best checkpoint）','',
        '| 模型 | 状态 | best epoch | mIoU% | glassIoU% | MAE | BF1% | FPR%↓ | FNR%↓ |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in rows:
        n=r['model'];v=models[n]['validation'] if n in models else {}
        values=[fmt(100*v[k]) if k in v else 'NOT_RUN' for k in ['mIoU','glass_IoU']]+[fmt(v.get('MAE'),5)]+[fmt(100*v[k]) if k in v else 'NOT_RUN' for k in ['boundary_F1','non_glass_FPR','glass_FNR']]
        text.append('| '+n+' | '+r['status']+' | '+str(r.get('best_epoch','—'))+' | '+' | '.join(values)+' |')
    text.extend(['','## 计算、延迟、显存','',
        '| 模型 | 推理参数 | 矩阵GFLOPs* | median ms | p95 ms | 推理峰值allocated MiB | 增量activation MiB |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for r in rows:
        text.append('| '+r['model']+' | '+str(r.get('inference_parameters','NOT_RUN'))+' | '+fmt(r['matrix_FLOPs_2MACs']/1e9 if r.get('matrix_FLOPs_2MACs') else None)+' | '+fmt(r.get('latency_median_ms'))+' | '+fmt(r.get('latency_p95_ms'))+' | '+fmt(r.get('inference_peak_allocated_MiB'))+' | '+fmt(r.get('activation_increment_MiB'))+' |')
    text.extend(['', '*矩阵计算量仅Conv2d+Linear+QK/AV的MAC×2，排除插值、归一化、池化和逐元素操作。延迟是同会话同步host计时的模型前向，包含归一化/模型内输出上采样，排除I/O及CPU原尺寸后处理；相同GPU/256/batch1/BF16/FP32参数、warm60、160次交替计时。显存在每个模型单独新进程测PyTorch峰值，不包括CUDA驱动和其他进程；reserved与训练batch8峰值另见JSON。历史FR29延迟不混用。',
        '', '## 训练数值曲线与收敛限制',''])
    for n,m in models.items():
        h=m.get('history') or [json.loads(x) for x in (cache/n/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if x]
        text.append(f"- {n}：best={m['best_epoch']}/40，最后5轮mIoU="+', '.join(fmt(100*r['validation']['mIoU']) for r in h[-5:])+'%；最后5轮train loss='+', '.join(fmt(r['mean_training_loss'],4) for r in h[-5:])+('。最佳轮位于末段，需警惕预算不足。' if m['best_epoch']>=36 else '。best早于末段本身不能证明充分收敛。'))
    text.extend(['','数值曲线保存在各模型epochs.jsonl和curves.csv；FR29历史原样复用。不做图、不因验证结果给单个候选追加预算。单种子不提供跨种子误差条或显著性结论。','', '## 比较结论与论文价值',''])
    if not full:text.append('**PENDING**：候选训练或同会话资源测量尚未全部结束，当前不能判断FR29相对外部模型是否有竞争力。FR29历史指标仅作已验证参照，BLOCKED/NOT_RUN项没有补造指标。')
    else:
        rv=models['FR29']['validation'];rl=models['FR29']['matched_latency'];complete=[]
        for n,m in models.items():
            if n=='FR29':continue
            v=m['validation'];d={k:100*(rv[k]-v[k]) for k in ['mIoU','glass_IoU','boundary_F1','non_glass_FPR','glass_FNR']}
            text.append(f"- FR29相对{n}：mIoU {d['mIoU']:+.3f}pp，玻璃IoU {d['glass_IoU']:+.3f}pp，BF1 {d['boundary_F1']:+.3f}pp，FPR {d['non_glass_FPR']:+.3f}pp，FNR {d['glass_FNR']:+.3f}pp（后两项负数有利）；参数差{278804-m['inference_parameters']:+d}。")
            if m['matched_latency'] and rl:
                dt=rl['median_ms']-m['matched_latency']['median_ms']
                text.append(f"  同会话median延迟差{dt:+.3f}ms。")
                dominated=(v['mIoU']>=rv['mIoU'] and m['inference_parameters']<=278804 and m['matched_latency']['median_ms']<=rl['median_ms'] and (v['mIoU']>rv['mIoU'] or m['inference_parameters']<278804 or dt>0))
                complete.append((n,dominated))
        if any(d for _,d in complete):text.append('**NO-GO（当前三维竞争力）**：至少一个已完成候选在mIoU/参数/实测median延迟上支配FR29。边界或漏检优势应单独描述，不能包装成综合领先。')
        elif len(complete)==len(cfg['runnable']):text.append('**有限GO（固定预算系统比较）**：本轮已完成候选中，FR29未在mIoU/参数/median延迟三维被支配。是否形成精度优势或资源优势，必须按上面差值分别叙述；这不等同于已证明全面超过外部方法。')
        else:text.append('**INCOMPLETE**：存在训练/测量失败，只能对完成模型作有条件比较。')
        text.append('论文对照价值：CGNet提供可靠作者架构的同量级参照，prefix隔离了部分预训练主干条件；受scratch/预训练、原生BN、辅助监督及40轮预算限制，尚不能作公平架构排行。两项BLOCKED使候选覆盖不足，且本轮是反复开发过的验证集；应把结果定位为资源受限条件下的开发证据，随后执行冻结的独立测试协议。')
        write(R/'evaluation_plan.frozen.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'status':'FROZEN_NOT_EXECUTED',
            'models':{n:{'best_epoch':m['best_epoch'],'checkpoint':str(Path(cfg['reference_path'])/'runs/FR29/best.pt' if n=='FR29' else cache/n/'best.pt')} for n,m in models.items()},
            'threshold':.5,'test_access_this_stage':False,'future_eval_requires_separate_authorization':True,
            'plan':'All completed models once on official GSD test under unchanged native-size postprocessing and six metrics; no test tuning, re-selection, extension or plots',
            'implementation_source_freeze':'execution.frozen.json'})
    text.extend(['','## 文件与恢复','',f"项目：{R}；大文件缓存：{cache}。每模型best.pt/last.pt/deployment.pt、epochs.jsonl、curves.csv、result.json、efficiency.json；正式控制台日志logs/run_*.log。断点续训执行 .\\run.ps1 -Mode run，按完整epoch恢复；不改变源文件或冻结配置。异常保留失败状态和已有文件，不自动绕过严格确定性或切换实现。",'', '当前没有官方测试结果；未下载大型数据；未覆盖历史实验；未绘图。'])
    (R/'FEASIBILITY_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(R/'summary.json',{'status':state['status'],'complete':full,'models':{n:{k:v for k,v in m.items() if k!='history'} for n,m in models.items()},'blocked':{n:m for n,m in audit['models'].items() if m['status']=='BLOCKED'},'official_test':'NOT_RUN'})
    print('Report updated: '+str(R/'FEASIBILITY_REPORT.md'))
if __name__=='__main__':main()
