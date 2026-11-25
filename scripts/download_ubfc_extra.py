"""Resolve official folder contents, then download one selected research recording."""
from pathlib import Path
import json,argparse,requests
from bs4 import BeautifulSoup
import gdown
p=argparse.ArgumentParser();p.add_argument('subject',choices=['subject3','subject4']);a=p.parse_args()
folders={'subject3':'1tJdI-E143hW0q0NHxmPPkLcr0UvZgeNz','subject4':'1vjXMUmi9CDWvAnrLNoteFNMoxU48Dhzt'}
folder=folders[a.subject];root=Path('data/public/ubfc')/a.subject;root.mkdir(parents=True,exist_ok=True)
r=requests.get('https://drive.google.com/drive/folders/'+folder,timeout=30);r.raise_for_status();(root/'official_index.html').write_text(r.text)
s=BeautifulSoup(r.text,'html.parser');files={}
for el in s.select('[data-id]'):
 ident=el.get('data-id');t=el.get_text(' ',strip=True)
 for name in ['ground_truth.txt','vid.avi']:
  if name in t and ident!='_gd':files[name]=ident
if len(files)!=2:raise ValueError('Cannot resolve official paired files')
(root/'download_provenance.json').write_text(json.dumps({'official_folder_id':folder,'files':files,'usage':'Official research dataset, author Agreement.xlsx marks this subject shareable; no redistribution'},indent=2))
for name,ident in files.items():
 target=root/name
 if not target.exists():
  print('Downloading',a.subject,name,flush=True)
  if not gdown.download(id=ident,output=str(target),quiet=False,use_cookies=False):raise RuntimeError('Download failed')
 print(name,target.stat().st_size,flush=True)
