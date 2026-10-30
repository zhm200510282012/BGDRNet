"""Visible sequential queue with atomic phase records and automatic final analysis."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, msvcrt, os, subprocess, sys, traceback
R=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
def stamp():return datetime.now(timezone.utc).isoformat()
def call(*args):
    command=[sys.executable,'-u',str(R/'runner.py'),*args]
    print('COMMAND: '+subprocess.list2cmdline(command),flush=True)
    subprocess.run(command,cwd=R,check=True)
def sources():
    f=read(R/'execution.frozen.json')
    assert read(R/'config.frozen.json')==f['config']
    assert all((R/k).read_text(encoding='utf-8')==v for k,v in f['sources'].items())
    assert (R/'manifest.csv').read_text(encoding='utf-8')==f['manifest']
    return f['config']
def audit_processes():
    ps=shutil_pwsh()
    code="Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    r=subprocess.run([ps,'-NoProfile','-Command',code],capture_output=True,text=True,encoding='utf-8')
    if r.returncode:raise RuntimeError('Cannot audit Python processes: '+r.stderr)
    procs=json.loads(r.stdout) if r.stdout.strip() else []
    if isinstance(procs,dict):procs=[procs]
    others=[p for p in procs if p['ProcessId']!=os.getpid()]
    if others:raise RuntimeError('RESOURCE_BLOCKED: other Python process(es); no automatic interference: '+str(others))
    return procs
def shutil_pwsh():
    import shutil
    return shutil.which('pwsh') or 'powershell.exe'

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','run']);a=p.parse_args()
    cfg=sources();cache=Path(cfg['cache'])
    with (R/'run.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            audit_processes()
            if a.mode=='smoke':call('smoke');subprocess.run([sys.executable,str(R/'analyze.py')],cwd=R,check=True);return
            assert read(R/'smoke_summary.json')['status']=='PASS'
            save(R/'launch_runtime.json',{'pid':os.getpid(),'started_utc':stamp(),'cwd':str(R),
                'command':subprocess.list2cmdline(sys.argv),'visible_console_required':True,'automatic_polling':False})
            failures={}
            for name in cfg['runnable']:
                sources();audit_processes()
                if not (cache/name/'result.json').exists():
                    save(R/'status.json',{'status':'TRAINING','model':name,'pid':os.getpid(),'utc':stamp(),'formal_training_started':True})
                    try:call('train','--model',name)
                    except subprocess.CalledProcessError:
                        failures[name]=traceback.format_exc();save(R/'failures.json',failures)
                        print('FAILED '+name+'; preserving checkpoint and moving to the next authorized independent baseline.',flush=True)
                        continue
                try:call('profile','--model',name)
                except subprocess.CalledProcessError:
                    failures[name+'_profile']=traceback.format_exc();save(R/'failures.json',failures)
                subprocess.run([sys.executable,str(R/'analyze.py')],cwd=R,check=True)
            audit_processes();call('profile','--model','FR29');call('benchmark')
            save(R/'status.json',{'status':'COMPLETE' if not failures else 'COMPLETE_WITH_FAILURES','utc':stamp(),
                'formal_training_started':True,'failures':list(failures),'official_test':'NOT_RUN'})
            subprocess.run([sys.executable,str(R/'analyze.py')],cwd=R,check=True)
            print('FINISHED: '+str(R/'FEASIBILITY_REPORT.md'),flush=True)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':
    try:main()
    except Exception:
        save(R/'status.json',{'status':'STOPPED_ERROR','utc':stamp(),'error':traceback.format_exc(),'official_test':'NOT_RUN'})
        raise
