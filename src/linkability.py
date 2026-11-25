"""Anonymous dataset-label matching metrics, not real-world identification."""
import numpy as np


def gallery_comparison(embedding,gallery,genuine_index):
    gallery=np.asarray(gallery,dtype=float);embedding=np.asarray(embedding,dtype=float).ravel()
    if gallery.ndim!=2 or gallery.shape[1]!=len(embedding) or len(gallery)<2:raise ValueError('Need at least two compatible templates')
    if not 0<=genuine_index<len(gallery) or not np.isfinite(gallery).all() or not np.isfinite(embedding).all():raise ValueError('Invalid gallery inputs')
    scores=gallery@embedding;own=float(scores[genuine_index]);others=np.delete(scores,genuine_index);impostor=float(np.max(others))
    return dict(genuine_cosine=own,max_impostor_cosine=impostor,margin=own-impostor,rank1_unique=bool(own>impostor),
                tied_top=bool(own==impostor),predicted_gallery_index=int(np.argmax(scores)))
