import importlib.util,json,sys
from pathlib import Path
source=Path(sys.argv[1]).resolve();sys.path.insert(0,str(source.parent))
spec=importlib.util.spec_from_file_location('method',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
first=json.loads(sys.stdin.readline());policy=module.initialize(first['initialize'])
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
    print(json.dumps(policy.step(json.loads(line)),separators=(',',':')),flush=True)
