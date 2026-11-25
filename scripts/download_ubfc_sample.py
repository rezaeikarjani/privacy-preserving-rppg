"""Download only the officially shared UBFC subject1 files; no redistribution.
Requires gdown (optional). Review docs/EXTERNAL_VALIDATION.md first.
"""
from pathlib import Path
import argparse
import gdown

p=argparse.ArgumentParser();p.add_argument('--destination',default='data/public/ubfc/subject1');a=p.parse_args()
root=Path(a.destination);root.mkdir(parents=True,exist_ok=True)
for name,identifier in [('ground_truth.txt','1q7LX_8Ggfl43ZmsFXMrUkV2hNnKr-yTE'),('vid.avi','1qBlkbaB8y3-KlWC_A61KsY42Wjo66ss4')]:
    target=root/name
    if target.exists():
        print('Existing file left unchanged:',target)
        continue
    result=gdown.download(id=identifier,output=str(target),quiet=False,use_cookies=False)
    if result is None:raise RuntimeError('Official download unavailable; no mirror substitution')
