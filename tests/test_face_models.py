import numpy as np
import pytest
from src.face_models import unit_vector,YuNetTracker


def test_embedding_normalization_preserves_cosine_under_scaling():
    x=np.arange(1,129,dtype=float)
    assert np.isclose(np.dot(unit_vector(x),unit_vector(5*x)),1.)


@pytest.mark.parametrize('x',[np.zeros(128),[1,np.nan],[np.inf,2]])
def test_invalid_embeddings_rejected(x):
    with pytest.raises(ValueError):unit_vector(x)


def test_detector_does_not_silently_select_among_multiple_faces():
    class Fake:
        def setInputSize(self,size):pass
        def detect(self,frame):return None,np.zeros((2,15),np.float32)
    tracker=YuNetTracker.__new__(YuNetTracker);tracker.detector=Fake()
    with pytest.raises(ValueError,match='multiple'):tracker.detect_one(np.zeros((50,50,3),np.uint8))
