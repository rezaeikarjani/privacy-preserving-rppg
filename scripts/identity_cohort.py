"""Small same-recording closed-set similarity/linkability demo; no anonymity guarantee."""
import json,hashlib
from pathlib import Path
import cv2,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.face_models import YuNetTracker,unit_vector
from src.linkability import gallery_comparison
from src.video import transform


def run():
    cfg=json.loads(Path('configs/identity_cohort.json').read_text());models=json.loads(Path('vendor/opencv_models/models.json').read_text())
    if models['revision']!=cfg['model_revision']:raise ValueError('Model revision changed')
    for name,info in models['files'].items():
        if hashlib.sha256((Path('vendor/opencv_models')/name).read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Model hash mismatch')
    out=Path('results/private/identity_cohort');out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);detector=YuNetTracker(score_threshold=cfg['detection_threshold']);recognizer=cv2.FaceRecognizerSF_create('vendor/opencv_models/sface.onnx','')
    gallery=[];sources={}
    def frame_at(subject,t):
        video=Path('data/public/ubfc')/subject/'vid.avi';cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);i=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,f=cap.read();cap.release()
        if not ok:raise ValueError('Cannot read '+subject+' at '+str(t))
        return i,f
    for subject in cfg['subjects']:
        traces=pd.read_csv(cfg['geometry'][subject]);sources[subject]=traces;features=[]
        for t in cfg['enrollment_times_s']:
            i,f=frame_at(subject,t);face=detector.detect_one(f);features.append(unit_vector(recognizer.feature(recognizer.alignCrop(f,face))))
        gallery.append(unit_vector(np.mean(features,axis=0)))
    gallery=np.asarray(gallery);rows=[]
    for identity,subject in enumerate(cfg['subjects']):
        for t in cfg['probe_times_s']:
            i,f=frame_at(subject,t)
            try:face=detector.detect_one(f)
            except ValueError as exc:
                for v in ['original','blur','pixelate','mask']:rows.append(dict(subject=subject,probe_time_s=t,variant=v,embedding_valid=False,original_detection_status=str(exc),rank1_unique=False,redetected=False))
                continue
            box=sources[subject].iloc[i][['x','y','w','h']].values
            for v in ['original','blur','pixelate','mask']:
                altered=transform(f,box,v,cfg['blur_sigma_fraction'],cfg['pixel_grid']);feature=unit_vector(recognizer.feature(recognizer.alignCrop(altered,face)));metrics=gallery_comparison(feature,gallery,identity)
                try:detector.detect_one(altered);redetected=True
                except ValueError:redetected=False
                rows.append(dict(subject=subject,probe_time_s=t,frame=i,variant=v,embedding_valid=True,original_detection_status='one_face',redetected=redetected,**metrics))
    table=pd.DataFrame(rows);table.to_csv(out/'probe_scores.csv',index=False);summaries=[]
    for v,s in table.groupby('variant',sort=False):
        valid=s[s.embedding_valid]
        summaries.append(dict(variant=v,attempted_probes=len(s),valid_embeddings=len(valid),original_detection_failures=len(s)-len(valid),
            mean_genuine_cosine=valid.genuine_cosine.mean(),mean_max_impostor_cosine=valid.max_impostor_cosine.mean(),
            mean_margin=valid.margin.mean(),rank1_unique_matches=int(valid.rank1_unique.sum()),rank1_fraction_attempted=valid.rank1_unique.sum()/len(s),
            rank1_fraction_valid=valid.rank1_unique.mean(),redetected_probes=int(s.redetected.sum())))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False)
    table.groupby(['subject','variant'],sort=False).agg(probes=('embedding_valid','size'),valid=('embedding_valid','sum'),matches=('rank1_unique','sum'),mean_genuine=('genuine_cosine','mean'),mean_impostor=('max_impostor_cosine','mean')).to_csv(out/'by_subject.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    x=np.arange(len(summary));axs[0].bar(x-.17,summary.mean_genuine_cosine,.34,label='Genuine');axs[0].bar(x+.17,summary.mean_max_impostor_cosine,.34,label='Strongest impostor');axs[0].legend();axs[0].set(ylabel='Mean cosine similarity',title='Original-landmark-assisted embeddings')
    axs[1].bar(x,summary.rank1_fraction_attempted);axs[1].set(ylim=(0,1),ylabel='Unique rank-1 matches / attempted probes',title='Three-person closed set; no rejection threshold')
    for ax in axs:ax.set_xticks(x,summary.variant)
    fig.suptitle('Small same-recording cohort: not independent-session privacy validation');fig.savefig(out/'cohort_linkability.png',dpi=150);plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(dict(config=cfg,models=models,opencv=cv2.__version__,gallery_subjects=len(gallery),
        embeddings_saved=False,face_images_saved=False,verification_threshold=None,
        caveat='Closed-set forced ranking among only three templates, same capture per identity. Probe correlations and mixed Haar/assisted geometry limit interpretation. No clinical or anonymity guarantee.'),indent=2))
    print(summary.to_string(index=False));print('\nBy subject:\n',pd.read_csv(out/'by_subject.csv').to_string(index=False))

if __name__=='__main__':run()
