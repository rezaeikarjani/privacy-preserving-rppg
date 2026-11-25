"""Official OpenCV Zoo CPU models, pinned by repository revision and local hashes."""
from pathlib import Path
import requests,json,hashlib
root=Path('vendor/opencv_models');root.mkdir(exist_ok=True,parents=True)
s=requests.Session();revision='48b10fa505aeb51fbd127fc60a5c660040e823b7'  # Pinned tested official revision
items={'yunet.onnx':'models/face_detection_yunet/face_detection_yunet_2023mar.onnx','sface.onnx':'models/face_recognition_sface/face_recognition_sface_2021dec.onnx','LICENSE_yunet':'models/face_detection_yunet/LICENSE','LICENSE_sface':'models/face_recognition_sface/LICENSE'}
manifest={'repository':'https://github.com/opencv/opencv_zoo','revision':revision,'files':{}}
for name,path in items.items():
 host='https://media.githubusercontent.com/media/opencv/opencv_zoo/' if name.endswith('.onnx') else 'https://raw.githubusercontent.com/opencv/opencv_zoo/'
 url=host+revision+'/'+path;res=s.get(url,timeout=60);res.raise_for_status()
 (root/name).write_bytes(res.content);manifest['files'][name]={'url':url,'bytes':len(res.content),'sha256':hashlib.sha256(res.content).hexdigest()}
 print(name,len(res.content),flush=True)
(root/'models.json').write_text(json.dumps(manifest,indent=2))
