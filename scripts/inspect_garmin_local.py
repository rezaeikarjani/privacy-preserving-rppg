"""Minimal FIT record reader for this local file, not a general FIT SDK.
Supports definitions, endian fields, developer-field skipping, compressed timestamps.
Unknown fields are skipped. Cross-checks HR/time against Garmin TCX.
"""
import struct,zipfile,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
p=Path('data/private');out=Path('results/private/garmin_reference');out.mkdir(parents=True,exist_ok=True)
z=zipfile.ZipFile(p/'25047062631.zip');data=z.read(z.namelist()[0]);header=data[0]
assert data[8:12]==b'.FIT'
end=header+int.from_bytes(data[4:8],'little');i=header;defs={};records=[];last=None
while i<end:
 h=data[i];i+=1
 if h&128:
  local=(h>>5)&3;compressed=h&31;definition=False
 else:
  local=h&15;compressed=None;definition=bool(h&64)
 if definition:
  reserved,arch=data[i:i+2];i+=2;order='little' if arch==0 else 'big'
  msg=int.from_bytes(data[i:i+2],order);i+=2;n=data[i];i+=1
  fields=[tuple(data[i+j*3:i+j*3+3]) for j in range(n)];i+=n*3
  dev=[]
  if h&32:
   n=data[i];i+=1;dev=[tuple(data[i+j*3:i+j*3+3]) for j in range(n)];i+=n*3
  defs[local]=(msg,order,fields,dev)
 else:
  msg,order,fields,dev=defs[local];values={}
  for num,size,base in fields:
   # compressed header replaces standard timestamp field
   if compressed is not None and num==253: continue
   b=data[i:i+size];i+=size
   if num in [253,3] and msg==20:
    v=int.from_bytes(b,order);values[num]=v if v!=(1<<(8*size))-1 else None
  for num,size,idx in dev:i+=size
  if compressed is not None:
   if last is None:raise ValueError('Compressed timestamp without base')
   ts=(last&~31)+compressed
   if compressed<(last&31):ts+=32
   values[253]=ts
  if values.get(253) is not None:last=values[253]
  if msg==20 and values.get(253) is not None and values.get(3) is not None:
   records.append((values[253],values[3]))
assert i==end,(i,end)
fit=pd.DataFrame(records,columns=['fit_timestamp','hr_bpm']);fit['utc']=pd.to_datetime(fit.fit_timestamp,unit='s',origin='1989-12-31',utc=True)
ns={'t':'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}
root=ET.parse(p/'activity_25047062631.tcx');activity_start=pd.Timestamp(root.find('.//t:Id',ns).text)
tcx=[]
for point in root.findall('.//t:Trackpoint',ns):
 v=point.find('t:HeartRateBpm/t:Value',ns)
 if v is not None:tcx.append((pd.Timestamp(point.find('t:Time',ns).text),int(v.text)))
lookup=dict(zip(fit.utc,fit.hr_bpm));matched=sum(lookup.get(t)==h for t,h in tcx)
fit['activity_time_s']=(fit.utc-activity_start).dt.total_seconds();fit['video_time_s']=fit.activity_time_s+6
fit[['activity_time_s','video_time_s','hr_bpm']].to_csv(out/'watch_samples.csv',index=False)
summary={'fit_record_count':len(fit),'tcx_record_count':len(tcx),'tcx_exact_matches_in_fit':matched,'hr_min':int(fit.hr_bpm.min()),'hr_max':int(fit.hr_bpm.max()),'median_gap_s':float(np.median(np.diff(fit.activity_time_s))),'max_gap_s':float(np.max(np.diff(fit.activity_time_s))),'video_start_offset_s':6,'offset_source':'user-reported spoken start; approximate','note':'Watch processed HR only; not raw PPG. No location fields exported. Minimal reader has no FIT CRC verification; corroborated against TCX.'}
(out/'inspection.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2));print(fit[['activity_time_s','video_time_s','hr_bpm']].to_string(index=False))
