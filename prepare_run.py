"""Create a NEW relocated run directory; never launch training."""
from pathlib import Path
import argparse,csv,json,shutil
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--job',required=True,choices=list(json.loads((R/'checkpoint_index.json').read_text())));p.add_argument('--gsd-root',type=Path,required=True);p.add_argument('--pretrained',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert a.pretrained.name=='mobilenet_v2-b0353104.pth' and a.pretrained.is_file(),'Use exact legacy pretrained checkpoint'
assert not a.out.exists(),'Refuse to overwrite an existing run'
rows=list(csv.DictReader((R/'splits/gsd_train_val.csv').open(encoding='utf-8')))
for row in rows:
 for k in ['image_path','mask_path']:
  row[k]=str((a.gsd_root/row[k]).resolve());assert Path(row[k]).is_file(),row[k]
src=R/'jobs'/a.job;a.out.mkdir(parents=True);(a.out/'logs').mkdir()
for q in src.glob('*.py'):shutil.copy2(q,a.out/q.name)
worker=a.out/'worker.py';s=worker.read_text(encoding='utf-8')
s=s.replace("Path('D:/SIVP/glass_light_mvp_20261025/model.py')","(R/'s0_original.py')")
s=s.replace("Path('D:/SIVP/glass_fr12_regional_context_20261028/logs/source_context_diagnostic.json')","(R/'fixed_smoke_ids.json')")
worker.write_text(s,encoding='utf-8')
for q in (R/'reference').iterdir():shutil.copy2(q,a.out/q.name)
cfg=json.loads((src/'config.template.json').read_text(encoding='utf-8'));cfg.update(weights=str(a.pretrained.resolve()),dataset_root=str(a.gsd_root.resolve()),output_root=str(a.out.resolve()))
(a.out/'config.frozen.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
with (a.out/'manifest.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Prepared only:',a.out)
print('Only configuration paths and historical smoke-reference paths relocated. Model/loss/optimizer unchanged.')
print('Future explicit execution: python worker.py --mode smoke; then python worker.py --mode train. Never run experiment.py --mode run: it retains an obsolete multi-variant main.')
