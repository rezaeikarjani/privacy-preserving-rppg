import importlib.util,json,platform,sys,subprocess
from pathlib import Path
names=['numpy','scipy','pandas','matplotlib','cv2','pyVHR','mediapipe','torch','pytest']
result={'platform':platform.platform(),'python':sys.version,'modules':{n:importlib.util.find_spec(n) is not None for n in names}}
result['git_network_probe']=subprocess.run(['git','ls-remote','https://github.com/ubicomplab/rPPG-Toolbox.git','HEAD'],capture_output=True,text=True,timeout=20).stderr.strip()
Path('results/environment.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
