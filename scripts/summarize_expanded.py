"""All selected outcomes; detector conditions kept separate, no best-run selection."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('results/private/expanded_validation');out.mkdir(parents=True,exist_ok=True)
rows=[];all_windows=[];status=[]
models=json.loads(Path('vendor/opencv_models/models.json').read_text())
for s in ['subject3','subject4']:
    d=Path('data/public/ubfc')/s
    status.append(dict(subject=s,video_available=(d/'vid.avi').exists(),reference_available=(d/'ground_truth.txt').exists(),video_sha256=hashlib.sha256((d/'vid.avi').read_bytes()).hexdigest()))
    for detector in ['haar','yunet']:
        p=Path('results/private')/f'external_ubfc_{s}_{detector}'
        if not (p/'reference_summary.csv').exists():
            status.append(dict(subject=s,detector=detector,status='failed_or_incomplete',log='results/'+s+'_'+detector+'_log.txt'));continue
        r=pd.read_csv(p/'reference_summary.csv');r.insert(0,'detector',detector);r.insert(0,'subject',s);rows.append(r)
        w=pd.read_csv(p/'reference_windows.csv');w.insert(0,'detector',detector);w.insert(0,'subject',s);all_windows.append(w)
        m=json.loads((p/'validation_manifest.json').read_text());m['evaluation_role']='New selected participant, settings fixed before outcomes'
        if detector=='yunet':m['detector_model']=models['files']['yunet.onnx'];m['detector_score_threshold']=.9
        (p/'validation_manifest.json').write_text(json.dumps(m,indent=2))
        status.append(dict(subject=s,detector=detector,status='completed',manual_assistance=False,total_windows=int(r.total_windows.max())))
r=pd.concat(rows,ignore_index=True);w=pd.concat(all_windows,ignore_index=True);r.to_csv(out/'per_subject_summary.csv',index=False);w.to_csv(out/'all_windows.csv',index=False)
conf=[]
for detector,s in w.groupby('detector',sort=False):
    s=s[(s.variant!='mask')&np.isfinite(s.abs_error_bpm)];truth=s.abs_error_bpm>5;flag=s.flagged
    tp=int((truth&flag).sum());fp=int((~truth&flag).sum());fn=int((truth&~flag).sum());tn=int((~truth&~flag).sum())
    conf.append(dict(detector=detector,correlated_window_variant_assessments=len(s),tp=tp,fp=fp,fn=fn,tn=tn,
                     precision=tp/(tp+fp) if tp+fp else float('nan'),recall=tp/(tp+fn) if tp+fn else float('nan'),
                     reference_ambiguity_assessments=int(s.contact_reference_quality_warning.sum())))
pd.DataFrame(conf).to_csv(out/'quality_confusion_descriptive.csv',index=False)
refs=w[(w.method=='POS')&(w.variant=='original')].copy();refs['device_minus_spectral_hr_bpm']=refs.contact_device_hr_bpm-refs.contact_ppg_hr_bpm
refs[['subject','detector','start_s','end_s','contact_ppg_hr_bpm','contact_device_hr_bpm','device_minus_spectral_hr_bpm','contact_reference_quality_warning']].to_csv(out/'reference_diagnostics.csv',index=False)
# 8 run/method rows, three transformation columns, all results displayed.
a=r[r.variant!='mask'].pivot(index=['subject','detector','method'],columns='variant',values='hr_mae_bpm')[['original','blur','pixelate']]
fig,ax=plt.subplots(figsize=(8,6),layout='constrained');im=ax.imshow(a.values,cmap='magma',vmin=0,vmax=max(1,float(np.nanmax(a.values))))
for i in range(len(a)):
    for j in range(3):ax.text(j,i,f'{a.values[i,j]:.2f}',ha='center',va='center',color='white' if a.values[i,j]<25 else 'black')
ax.set_xticks(range(3),a.columns);ax.set_yticks(range(len(a)),[' / '.join(x) for x in a.index]);ax.set_title('New participants: MAE vs fixed contact-PPG spectral HR\nSubject3 reference is itself uncertain; conditions remain separate')
fig.colorbar(im,ax=ax,label='BPM');fig.savefig(out/'expanded_mae.png',dpi=150);plt.close(fig)
manifest={'selection':json.loads(Path('configs/expanded_validation.json').read_text()),'status':status,'source':'User placed selected official videos in prescribed local folders after earlier quota failure','scope':'Only two new participants; 3 distinct overlapping-window intervals, 18 correlated method/variant assessments per detector, not 18 independent participants','reference_caution':'Subject3 contact PPG spectral HR differs from device-reported HR and first spectral-reference window is ambiguous. No reference was silently substituted or removed. Confusion labels are conditional on this reference, not definitive physiological truth.'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(r[r.variant!='mask'][['subject','detector','method','variant','evaluable_windows','hr_mae_bpm','flagged_windows']].to_string(index=False));print('\nDescriptive warning counts:\n',pd.DataFrame(conf).to_string(index=False))
