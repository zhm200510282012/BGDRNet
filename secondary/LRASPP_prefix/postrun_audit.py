"""Read-only checkpoint/history consistency audit, no fitting or dataset evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,math
import torch
import runner

R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    runner.check_sources();cfg=runner.CFG;cache=Path(cfg['cache'])
    assert read(R/'status.json')['status']=='COMPLETE'
    summary=read(R/'summary.json');results={};areas=[]
    for name in ['FR29',*cfg['runnable']]:
        root=Path(cfg['reference_path'])/'runs/FR29' if name=='FR29' else cache/name
        history=[json.loads(x) for x in (root/'epochs.jsonl').read_text(encoding='utf-8').splitlines() if x]
        assert [r['epoch'] for r in history]==list(range(1,41))
        assert all(r['global_step']==271*r['epoch'] for r in history)
        assert all(math.isfinite(r['mean_training_loss']) for r in history)
        best=max(history,key=lambda r:(r['validation']['mIoU'],-r['epoch']))
        ck=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
        last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        assert ck['epoch']==best['epoch'] and ck['validation']==best['validation']==summary['models'][name]['validation']
        assert ck['manifest']==last['manifest']==runner.FROZEN['manifest']
        assert last['epoch']==40 and last['scheduler']['last_epoch']==40
        assert last['optimizer']['param_groups'][0]['lr']==0
        assert all(float(v['step'])==10840 for v in last['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in ck['model'].values() if v.is_floating_point())
        net=runner.model(name)
        deployed=torch.load(root/'deployment.pt',map_location='cpu',weights_only=False)
        net.load_state_dict(deployed['model'],strict=True)
        params=sum(p.numel() for p in net.parameters())
        assert params==summary['models'][name]['inference_parameters']
        expect={k:ck['model'][k] for k in net.state_dict()}
        assert expect.keys()==deployed['model'].keys()
        assert all(torch.equal(expect[k],deployed['model'][k]) for k in expect)
        if name!='FR29':
            runner.verify(ck,name);runner.verify(last,name)
            assert last['record']==history[-1]
            curves=list(csv.DictReader((root/'curves.csv').open(encoding='utf-8')))
            assert len(curves)==40
            assert all(float(row['mIoU'])==h['validation']['mIoU'] for row,h in zip(curves,history))
        val=best['validation'];gt_glass=val['TP']+val['FN'];gt_bg=val['TN']+val['FP'];areas.append((gt_glass,gt_bg))
        gi=val['TP']/(val['TP']+val['FP']+val['FN']);bi=val['TN']/(val['TN']+val['FP']+val['FN'])
        assert abs(gi-val['glass_IoU'])<1e-12 and abs((gi+bi)/2-val['mIoU'])<1e-12
        assert abs(val['FN']/gt_glass-val['glass_FNR'])<1e-12
        assert abs(val['FP']/gt_bg-val['non_glass_FPR'])<1e-12
        assert val['images']==542
        l=read(R/'matched_latency.json');assert len(l['raw_ms'][name])==160
        results[name]={'status':'PASS','best_epoch':best['epoch'],'epochs':40,'optimizer_steps':10840,
            'inference_parameters':params,'full_checkpoint_resume_state_present':True,'deployment_matches_best':True,
            'same_manifest_config_source_verified':True,'curve_matches_checkpoint':True,
            'gt_glass_pixels':gt_glass,'gt_background_pixels':gt_bg,
            'first20_best_mIoU':max(h['validation']['mIoU'] for h in history[:20]),
            'last20_best_mIoU':max(h['validation']['mIoU'] for h in history[20:]),
            'best_minus_epoch40_pp':100*(best['validation']['mIoU']-history[-1]['validation']['mIoU'])}
        del ck,last,net,deployed
    assert len(set(areas))==1
    f=read(R/'evaluation_plan.frozen.json');assert f['status']=='FROZEN_NOT_EXECUTED' and not f['test_access_this_stage']
    result={'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),'checks':results,
        'official_test':'NOT_RUN','gpu_inference_or_fitting_this_audit':False,'same_gt_pixel_totals':True,
        'source_freeze_unchanged':True,'future_evaluation_plan_frozen':True}
    (R/'completed_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
