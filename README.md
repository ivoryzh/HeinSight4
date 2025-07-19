[![PyPI - Version](https://img.shields.io/pypi/v/heinsight)](https://pypi.org/project/heinsight/)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14630321.svg)](https://doi.org/10.5281/zenodo.14630321)
[![YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?logo=youtube)](https://youtube.com/shorts/u9_i0PKJr4w)
[![Hugging Face](https://img.shields.io/badge/Demo-HuggingFace-blue?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/AccelerationConsortium/HeinSight-Demo)


# HeinSight4.0 – A Computer Vision System for Dynamic Monitoring of Chemical Experiments
HeinSight4.0 is a computer vision system designed for real-time monitoring of chemical behavior. It detects and classifies chemical phases (air, liquid, solid) within vessels, enabling automated observation of common experimental behaviors such as dissolution, melting, suspension, mixing, settling, and more. It also extracts additional visual cues like turbidity and color through image analysis.

**This model was tested on chemistry within vials and EasyMax reactor.**
> 💡 **Installation is now available with**  
> `pip install heinsight`  
>  
> 👉 Try it out with the [HeinSight Demo app](https://huggingface.co/spaces/ivoryzhang/HeinSight-Demo).

## Table of Contents
- [How It Works?](#how-it-works)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Example GUI](#example-gui)
- [Acknowledgements](#acknowledgements)


## How It Works?
HeinSight4.0 employs a hierarchical detection approach by training two separate models **(Figure 1)**:
- Vessel Detection Model: Identifies transparent laboratory equipment (e.g., reactors and vessels) and marks them as "vessels."
- Chemical Detection Model: Detects chemical artifacts and phases within the identified vessels. The model classifies chemical phases into five categories, as outlined in **Table 1**.

The output of the vessel detection model serves as input for the chemical detection model, enabling phase-specific analysis. Both models were fine-tuned from YOLOv8 pretrained on the COCO dataset and adapted to our customized datasets.
Details on models training can be found here: https://zenodo.org/records/15605098. 

![](https://gitlab.com/heingroup/heinsight4.0/-/raw/main/docs/model_method.png)
**Figure 1. hierarchical detection approach of HeinSight4.0.** 



**Table 1. Classes names for chemical detection model**. 
![](https://gitlab.com/heingroup/heinsight4.0/-/raw/main/docs/classes.png)


## Datasets
### Vessel Dataset
Composed of 6493 images from the HeinSight3.0 dataset combined with additional images of reactors and vessels to expand detection capabilities across various laboratory setups.
	
### Chemical Dataset
Includes 3801 images captured from video footage of dynamic chemical experiments.
Features diverse scenarios:
* Varied background lighting
* A range of colored liquids and compounds
* Different solid forms and behaviors in liquid environments
This dataset enables monitoring of key experimental behaviors, including dissolution, melting, mixing, settling, and others, to address complex experimental conditions. A representative set of images is shown in **Figure 2**.

Dataset can be accessed at https://zenodo.org/records/14630321


![](https://gitlab.com/heingroup/heinsight4.0/-/raw/main/docs/dataset.png)

**Figure 2. Overview of diverse images in the training dataset used for the chemical detection model**

## Installation
The script was developed on Windows and tested on a Raspberry Pi 5. Python versions should be flexible and align with the requirements of Ultralytics (Python>=3.8 environment with PyTorch>=1.8).

### For Users

You can install `heinsight` directly from PyPI:

```bash
pip install heinsight
```
### For Developers
If you want to contribute to the project, you can clone the repository and install the dependencies from requirements.txt:
```commandline
git clone https://gitlab.com/heingroup/heinsight4.0.git
cd heinsight4.0
pip install -r requirements.txt
cd heinsight
```
### Enable CUDA
Note that PyTorch installation can be different when using a Nvidia GPU, check the [PyTorch](https://pytorch.org/) page for more detail.  
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### Installation on a Raspberry Pi

Depending on the model sizes, and Pi model (pi 4 or 5), it might need to use headless openCV to avoid GUI conflict. Check out the [Stream](#stream) section for real time monitoring on a Pi device.
```
pip uninstall opencv-python
pip install opencv-python-headless
```
Note that `picamera2` installation uses `apt`. Check formal installation guide for more details
```
sudo apt install -y python3-picamera2
```

## Usage
### Quickstart 
Once installed, you can start the server with the heinsight-server command:

```Bash
heinsight-server
```
and use --help for arguments doc
```Bash
heinsight-server --help
```
### Image mode usage with [demo image](https://gitlab.com/heingroup/heinsight4.0/-/blob/main/examples/demo.png)
```Python
from heinsight.heinsight import HeinSight
heinsight = HeinSight(vial_model_path=r"models/best_vessel.pt",
                      contents_model_path=r"models/best_content.pt", )
heinsight.run("path/to/img.png")
```
Output: heinsigh_output/[output.png](https://gitlab.com/heingroup/heinsight4.0/-/blob/main/examples/demo_output.png)

### Video analysis 
Output: 
* **heinsigh_output/output.mkv**: analysis output
* **heinsigh_output/output_per_phase.csv**: turbidity and color (overall and per phase) over time
* **heinsigh_output/output_raw.csv**: turbidity per row over time
```python
heinsight.run("path/to/video.mp4")
```

### Realtime monitoring with a webcam
Output: Video analysis output + raw video capture
```python
# realtime analysis example
heinsight.run(0)
```

### Other arguments
```python
heinsight.run("path/to/video.mp4", 
              save_directory="new_folder",  # save to other path
              output_name="filename",       # save with other base filename
              fps=5,                        # capture frame rate, only available with webcam
              res=(1920, 1080))             # capture resolution, only available with webcam
```

### Stream
Stream with a FastAPI app, in [stream.py](https://gitlab.com/heingroup/heinsight4.0/-/blob/main/heinsight/stream.py)

```commandline
pip install "fastapi[standard]"
cd heinsight
fastapi run stream.py
```
### API Endpoints
* `GET /docs`: View the interactive API documentation (Swagger UI).
* `POST /start`: Start the monitoring.
  * Body:
  ```json
  {
    "video_source": 0,
    "frame_rate": 30,
    "res": [1920, 1080]
  }
    ```
* `GET /stop`: Stop the monitoring.

* `GET /frame`: Get the latest processed video frame for streaming.

* `GET /data`: Get the collected data.

* `GET /current_status`: Get the most recent status and data point.

## Integration
For integration usage, we recommend to use heinsight_api
```python
from heinsight.heinsight_api import HeinsightAPI

heinsight = HeinsightAPI("http://localhost:8000", source=0, res=(1920, 1080))

# check is the sample homogeneous
heinsight.homo()

# check the volume
heinsight.volume_1()
```
## Example GUI
A sample HTML dashboard is provided in examples/sample_gui.html to demonstrate how to interact with the API.



## Acknowledgements
Rama El-khawaldeh, Ivory Zhang, Ryan Corkery
