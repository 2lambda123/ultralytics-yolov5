#!/bin/bash
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Download latest models from https://github.com/ultralytics/yolov5/releases
# Example usage: bash path/to/download_weights.sh
# parent
# └── yolov5
#     ├── yolov5s.pt  ← downloads here
#     ├── yolov5m.pt
#     └── ...

python - <<EOF
import os
from utils.downloads import attempt_download

models = ['n', 's', 'm', 'l', 'x']
if os.environ.get('YOLOV5_DOWNLOAD_LARGE_IMAGE_MODELS'):
    models.extend([i + '6' for i in models])

for x in models:
    attempt_download(f'yolov5{x}.pt')

EOF
