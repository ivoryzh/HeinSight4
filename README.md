# HeinSight 2.5

## Description
HeinSight



## Installation
The script was developed on Window, and tested on Windows and a Raspberry Pi 5. Python versions should be flexible and align with the requirements of Ultralytics
(Python>=3.8 environment with PyTorch>=1.8). 
```commandline
git clone https://gitlab.com/heingroup/heinsight2.5.git
cd heinsight2.5
pip install -r requirements.txt
```
### Enable CUDA
Note that PyTorch installation can be different when using a Nvidia GPU, check the [PyTorch](https://pytorch.org/) page for more detail.  
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage
```python
from heinsight.heinsight import HeinSight
heinsight = HeinSight(vial_model_path="models/labpic.pt",
                      contents_model_path=r"models/best_train5_yolov8_ez_20240402.pt")

# realtime analysis example
heinsight.run(0)
```


### Stream
Stream with a FastAPI app, in stream.py

```python
from heinsight.heinsight import HeinSight

...

heinsight = HeinSight(vial_model_path="models/labpic.pt",
                      contents_model_path=r"models/best_train5_yolov8_ez_20240402.pt")
source = 0

...
```
```commandline
pip install "fastapi[standard]"
fastapi run stream.py
```
### URLs
* Start monitoring:   localhost:8000/start 
* Stop monitoring:    localhost:8000/stop 
* Analysis output:    localhost:8000/frame