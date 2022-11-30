# Paper Reproduction: OW-DETR: Open-world Detection Transformer (CVPR 2022)

The release of souce code (https://github.com/akshitac8/OW-DETR) carries out experienments in mulitple tasks such as incremental object detection and 
semantic segmentation, and designed a series of optional architechure. Besides, there are some errors in the file structure and Readme.Therefore, it 
is too redundancy to reproduce a owdetr for pure object detection. 

Here, I reproduce the one-stage OW-DETR with box refinement only for incremental object detection.
![image](https://user-images.githubusercontent.com/46144673/204724170-03bd5515-260c-4d6e-9a4a-98f7d0dd62df.png)

## Dataset Preparation
In this project, the COCO 2017 is used to train and evaluate the model. 
You can download the dataset using following command:
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
Unzip the dataset and move to corresponding directory:
```
unzip train2017.zip -d ./OW-DETR/Data/coco/.
unzip val2017.zip ./OW-DETR/Data/coco/.
unzip annotations_trainval2017.zip ./OW-DETR/Data/coco/.
mv train2017 val2017 ./OW-DETR/Data/OWDETR/VOC2007/JPEGimages/.
```
Use the code coco2voc.py for converting json annotations to xml files. THe xml fiels and image details will be respectively placed at: 
```./OW-DETR/Data/OWDETR/VOC2007/Annotations``` and ``` ./OW-DETR/Data/OWDETR/VOC2007/ImageSets/Main/train.txt```.
The command is as following:
```
cd ./OW-DETR/Datasets
python coco2voc.py
```
As this OW-DETR aims at incremental object detection, the dataset is splited into 4 parts for 4 training stages, each of which introduces new seen classes and continues to train on the basis of the pretrained model from previous stage. At each stage, the unknown objects are forward to an oracle to label new classes and added to the known classes, which can be mimicked by introducing new seen classes at next stage.
![image](https://user-images.githubusercontent.com/46144673/204722902-9e5e4616-eeb3-409f-9e3f-0be747646075.png)
The split txt file is already at:  ```./OWDETR/Data/OWDETR/VOC2007/ImageSets``` .
## Enviroment Preparation
Check if conda exists:
```
conda info --env
```
If conda does not exist, install conda and configure using following command:
```
!wget -qO ac.sh https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
!bash ./ac.sh -b
```
```
import os
import sys
os.environ['PATH'] = '/root/anaconda3/bin:' + os.environ['PATH']
_ = (sys.path.append("/usr/local/lib/python3.8/site-packages"))
```
Create virtual environment and install requirements using following command:
```
conda create -n owdetr python=3.7 pip
conda activate owdetr
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
## Compiling CUDA operators
```
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```
## Training
As its nature of Incremental Object Detection, the training is split to 4 stages. The current stage would introduce new seen classes and continue to train
the pretrain model out of previous stage. The configuration for hyparameters of 4 stages is at: ```./OW-DETR/configs/```.

If train using 1 GPU, you can directly use command:
```
cd ./OW-DETR
run_uniGPU.sh
```
If train using mulitple GPU in a distributed environment, you can use command:
```
cd ./OW-DETR
run.sh
```
This shell first configures and calls run_dist_launch.sh, and run_dist_launch.sh calls launch.py to launch distributed computation, where training script is asigned to multiple processes at multiple nodes.
## Evaluation
Use following command:
```
cd ./OW-DETR
run_eval.sh
```
## Citation
```
@inproceedings{gupta2021ow,
    title={OW-DETR: Open-world Detection Transformer}, 
    author={Gupta, Akshita and Narayan, Sanath and Joseph, KJ and 
    Khan, Salman and Khan, Fahad Shahbaz and Shah, Mubarak},
    booktitle={CVPR},
    year={2022}
}
```
