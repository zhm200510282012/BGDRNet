"""Standard-library-only recomputation; never imports torch or opens images."""
from pathlib import Path
import csv,json,statistics
R=Path(__file__).resolve().parent
K=['mIoU','glass_IoU','MAE','boundary_F1','non_glass_FPR','glass_FNR']
def aggregate(rows):
 tp,fp,tn,fn=[sum(int(r[k]) for r in rows) for k in ['TP','FP','TN','FN']];n=tp+fp+tn+fn
 return dict(mIoU=(tp/(tp+fp+fn)+tn/(tn+fp+fn))/2,glass_IoU=tp/(tp+fp+fn),MAE=sum(float(r['absolute_error_sum']) for r in rows)/n,boundary_F1=statistics.mean(float(r['boundary_F1']) for r in rows),non_glass_FPR=fp/(fp+tn),glass_FNR=fn/(fn+tp))
expected=json.loads((R/'evidence/summary.json').read_text(encoding='utf-8'))
out={};endpoints=0
for dataset,count in [('GSD_test',813),('GEGD2',950)]:
 metrics={}
 for model in ['S0','FR29']:
  for seed in [42,43,44]:
   key=f'{model}_seed{seed}';folder=R/'evaluation'/dataset/key;rows=[]
   for p in sorted(folder.glob('part_*.csv')):rows+=list(csv.DictReader(p.open(encoding='utf-8-sig')))
   manifest=list(csv.DictReader((R/'splits'/('gsd_test.csv' if dataset=='GSD_test' else 'gegd2.csv')).open(encoding='utf-8-sig')))
   assert len(rows)==count and [r['sample_id'] for r in rows]==[r['sample_id'] for r in manifest]
   metrics[key]=aggregate(rows)
   for k in K:assert abs(metrics[key][k]-expected['per_seed'][key][dataset][k])<1e-10,(key,dataset,k)
   endpoints+=1
 out[dataset]={}
 for k in K:
  values=[metrics[f'FR29_seed{s}'][k]-metrics[f'S0_seed{s}'][k] for s in [42,43,44]]
  out[dataset][k]={'values':values,'mean':statistics.mean(values),'sample_sd_ddof1':statistics.stdev(values)}
  for field in ['mean','sample_sd_ddof1']:assert abs(out[dataset][k][field]-expected['paired'][dataset][k][field])<1e-10
print(json.dumps({'status':'PASS','endpoints':endpoints,'paired':out,'no_training_or_inference':True},indent=2))
