"""Fail explicitly if pyVHR is unavailable. Save separate result directory."""
import argparse,json
from pathlib import Path
from src.run import run
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());cfg['backend']='pyvhr';cfg['output']+='_pyvhr'
target=Path('configs/pyvhr_generated.json');target.write_text(json.dumps(cfg,indent=2))
run(target)
