"""One-person SFace similarity experiment; never publishes face images or embeddings."""
from pathlib import Path
import json,hashlib
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.face_models import YuNetTracker,unit_vector
from src.video import transform


def run():
    cfg=json.loads(Path('configs/identity_demo.json').read_text());models=json.loads(Path('vendor/opencv_models/models.json').read_text())
    for name,meta in models['files'].items():
        if hashlib.sha256((Path('vendor/opencv_models')/name).read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Model bytes changed')
    p=Path('results/private/identity_demo');p.mkdir(parents=True,exist_ok=True)
    video='data/public/ubfc/subject1/vid.avi';cap=cv2.VideoCapture(video);fps=cap.get(cv2.CAP_PROP_FPS)
    tracker=YuNetTracker(score_threshold=cfg['detection_score_threshold']);recognizer=cv2.FaceRecognizerSF_create('vendor/opencv_models/sface.onnx','');cv2.setNumThreads(1)
    # Use the same transformation box geometry as the existing ROI-assisted physiological experiment.
    tracks=pd.read_csv('results/private/external_ubfc_subject1_roi/tracking.csv')
    def read(t):
        index=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,f=cap.read()
        if not ok:raise ValueError('Cannot decode requested frame')
        return index,f
    enrollment=[]
    for t in cfg['enrollment_times_s']:
        i,frame=read(t);face=tracker.detect_one(frame)
        enrollment.append(unit_vector(recognizer.feature(recognizer.alignCrop(frame,face))))
    gallery=unit_vector(np.mean(enrollment,axis=0));rows=[]
    for t in cfg['probe_times_s']:
        i,frame=read(t);face=tracker.detect_one(frame);box=tracks.iloc[i][['x','y','w','h']].values
        for variant in ['original','blur','pixelate','mask']:
            modified=transform(frame,box,variant,cfg['blur_sigma_fraction'],cfg['pixel_grid'])
            # Always compute assisted embeddings, even if an end-to-end detector fails.
            feature=unit_vector(recognizer.feature(recognizer.alignCrop(modified,face)))
            try:
                newface=tracker.detect_one(modified);redetected=True
                end_to_end_feature=unit_vector(recognizer.feature(recognizer.alignCrop(modified,newface)))
                end_to_end_cosine=float(np.dot(gallery,end_to_end_feature));error=''
            except ValueError as exc:redetected=False;end_to_end_cosine=float('nan');error=str(exc)
            rows.append(dict(video_time_s=t,frame=i,variant=variant,assisted_cosine=float(np.dot(gallery,feature)),redetected=redetected,
                             end_to_end_cosine=end_to_end_cosine,detection_status=error or 'one_face'))
    cap.release();table=pd.DataFrame(rows);table.to_csv(p/'probe_scores.csv',index=False)
    summary=table.groupby('variant',sort=False).agg(probes=('assisted_cosine','size'),mean_assisted_cosine=('assisted_cosine','mean'),
        min_assisted_cosine=('assisted_cosine','min'),max_assisted_cosine=('assisted_cosine','max'),redetected_probes=('redetected','sum'),
        mean_end_to_end_cosine=('end_to_end_cosine','mean')).reset_index();summary.to_csv(p/'summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for v in ['original','blur','pixelate','mask']:
        s=table[table.variant==v];ax.plot(s.video_time_s,s.assisted_cosine,'o-',label=v)
    ax.set(xlabel='Probe time in recording (s)',ylabel='Cosine similarity to original enrollment',ylim=(-1,1),
        title='Single-person, same-recording embedding demo; original-landmark assistance')
    ax.legend(ncol=4);fig.savefig(p/'embedding_similarity.png',dpi=150);plt.close(fig)
    (p/'manifest.json').write_text(json.dumps(dict(config=cfg,models=models,opencv=cv2.__version__,embedding_dimension=len(gallery),
        inference='OpenCV CPU; original-landmark alignment and separate transformed-frame re-detection',
        gallery='Mean of normalized embeddings from original frames at 5,10,15s; normalized again',
        probes='30,35,40s; no identical enrollment/probe frames; same person and capture session',
        no_embeddings_saved=True,no_faces_exported=True,verification_threshold=None,
        limitation='No impostors, independent sessions or calibrated verification threshold; cannot estimate recognition accuracy, false-match rates, or guarantee anonymity.'),indent=2))
    print(summary.to_string(index=False));return summary

if __name__=='__main__':run()
