"""Visible current-only development subprocess; not the sealed final grader."""
import argparse,json,math,os,resource,selectors,subprocess,sys,time
from pathlib import Path
from trusted_core import State

def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(3,3));resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,64*1024**2))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def one(ep,source,root):
    start=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);state=State(ep['public']);buffer=b''
    p=subprocess.Popen([sys.executable,'-B',str(root/'worker.py'),str(source)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        preexec_fn=limits,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    select=selectors.DefaultSelector();select.register(p.stdout,selectors.EVENT_READ)
    def read():
        nonlocal buffer
        while b'\n' not in buffer:
            if not select.select(max(0,20-(time.monotonic()-start))):raise TimeoutError('response deadline')
            data=os.read(p.stdout.fileno(),65536)
            if not data:raise ValueError('worker closed early')
            buffer+=data
            if len(buffer)>64*1024**2:raise ValueError('output limit')
        line,buffer=buffer.split(b'\n',1);return json.loads(line)
    try:
        p.stdin.write(json.dumps({'initialize':ep['public']}).encode()+b'\n');p.stdin.flush()
        if read()!={'ready':True}:raise ValueError('initialization')
        for target in ep['requests']:
            p.stdin.write(json.dumps({'target':target,'state':state.snapshot()}).encode()+b'\n');p.stdin.flush()
            state.serve(target,read())
        p.stdin.close();p.wait(timeout=5)
        after=resource.getrusage(resource.RUSAGE_CHILDREN);cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
        if p.returncode or cpu>3:raise ValueError('CPU/exit contract')
        return dict(id=ep['id'],valid=True,**state.metrics(),cpu_seconds=cpu)
    except Exception as e:
        p.kill();p.wait();return dict(id=ep['id'],valid=False,error=str(e))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','validation'],default='validation')
    ap.add_argument('--solver',default='methods/main/solver.py');ap.add_argument('--output',default='/tmp/cache_selfcheck.json')
    a=ap.parse_args();root=Path(__file__).resolve().parent;source=Path(a.solver)
    if not source.is_absolute():source=root/source
    episodes=json.loads((root/'visible'/f'{a.split}.json').read_text());rows=[one(e,source,root) for e in episodes]
    Path(a.output).write_text(json.dumps(rows,indent=2));valid=all(r['valid'] for r in rows)
    print(json.dumps(dict(episodes=len(rows),valid=sum(r['valid'] for r in rows),all_valid=valid,
        total_cost=sum(r['cost'] for r in rows) if valid else None,output=a.output),indent=2))

if __name__=='__main__':main()
