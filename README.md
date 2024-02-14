# CA-PatchCore

**This repository is implementation of CA-PatchCore on Co-occurrence Anomaly Detection Dataset (CAD-SD).**

## Installation
Donwload "Co-occurrence Anomaly Detection Screw Dataset (CAD-SD)" to ./dataset from this [[Link](https://drive.google.com/drive/folders/1yeampzTiB4uoTmmqIZkeCdMIXGujl3cU?usp=sharing)].  

###environment
~~~
python==3.10.12
torch==2.0.1
torchvision==0.15.2
~~~

Install packages with:
~~~
pip install -r requirements.txt
~~~

Install PyDenceCRF with:
~~~
pip install cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
~~~
(Reference:https://github.com/lucasb-eyer/pydensecrf)

## Usage
Run with:
~~~
python main.py 
~~~

**Click [here(https://github.com/IshidaKengo/CA-PatchCore-_on_MVTec-LOCO-AD)] for implementation in MVTec LOCO AD.**
