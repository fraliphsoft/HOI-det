## Instance of Interest Detection

This repo is code for "Human Object Interaction Detection via Multi-level Conditioned Network" (ICMR'20).

<b>Note</b> that the 'master' branch is only in support of python 2.7 + pytorch 0.4.0.
If latest runtime environment is required, please checkout to 'pytorch1.x' branch (varified by python 3.8 + pytorch 1.9.0 + cuda 11.1):

```
git clone https://github.com/fraliphsoft/HOI-det.git -b pytorch1.x
or
git clone https://github.com/fraliphsoft/HOI-det.git
cd HOI-det
git checkout pytorch1.x
```

## Requirements
backports.functools-lru-cache==1.6.1
certifi==2019.11.28
cffi==1.14.0
cycler==0.10.0
Cython==0.29.16
easydict==1.9
kiwisolver==1.1.0
matplotlib==2.2.5
numpy==1.16.6
opencv-python==4.2.0.32
Pillow==6.2.2
protobuf==3.11.3
pycparser==2.20
pyparsing==2.4.6
python-dateutil==2.8.1
pytz==2019.3
PyYAML==5.3.1
scipy==1.2.3
six==1.14.0
subprocess32==3.5.4
tensorboardX==2.0
tensorflow==1.1.0
torch==0.4.0
torchvision==0.2.1
tqdm==4.44.1

## Installation
#### Install Environment
```
#---optional---
conda create --name <env-name> python=2.7
conda activate <env-name>
```
```
pip install -r requirements.txt
```
#### Install Matlab
We adopt the implementation of evaluation functions in HICO dataset from [TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network)(actually from [iCAN](https://github.com/vt-vl-lab/iCAN) or former), so matlab runtime environment is needed.
There are no restrictions on versions, you can download matlab from its official website and install according to the tutorial.

Note:
If 'Parallel Computing Toolbox' is not available in your matlab license, some changes should be made to run the evaluation code of HICO-Det:
- in 'Generate_detection.m' file:

```
delete line 285: delete(gcp('nocreate'))
```

- in 'ecal_one.m' file:

```
1. delete lines 119-123:
	if ~exist('pool_size','var')
		poolobj = parpool();
	else
		poolobj = parpool(pool_size);
	end  
2. alter line 132: parfor -> for
3. delete line 192: delete(poolobj);
```

#### Install Libraries
```
1. cd ./lib/
2. sh make.sh # if there is already './lib/pycocotools/_mask.so' exsits, delete it before step 3
3. cd ./model/nms
4. sh make.sh
```

## Prepare Data

---
<b>
We have integrated everything of data files from different sources into a specific repository folders for easier access:
https://zenodo.org/record/5823557
You can download all required first, and then follow the setps below.
</b>

---

#### 1. Download Datasets
We perform our experiments in 'vcoco' and 'hico(v2016)' dataset. We follow the setting of [TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network) as our original dataset, and then use [WSHP](https://github.com/MVIG-SJTU/WSHP) to extract human body part regions in hico and vcoco dataset respectively.
Considering that the body part regions data is too large(>100GB for test and >500GB for training) while the code runs fast, we provide prepared dataset except human body part regions:
- [vcoco](https://drive.google.com/file/d/1NlyrqhtYUlQCkLXJzM036AqCeo8aVjs9/view?usp=sharing): https://drive.google.com/file/d/1NlyrqhtYUlQCkLXJzM036AqCeo8aVjs9/view?usp=sharing
- [hico](https://drive.google.com/file/d/1KFXWUt6lCvXGflpq6tvrfqtPES6VoPbk/view?usp=sharing): https://drive.google.com/file/d/1KFXWUt6lCvXGflpq6tvrfqtPES6VoPbk/view?usp=sharing

You can download the datasets and extract them to './data/hico/' and './data/vcoco/' respectively.
#### 2. Calculate Human Body Part Regions
We provide the fine-tuned [WSHP](https://github.com/MVIG-SJTU/WSHP) code in './WSHP', you can run it following these steps:
```
1. # download pretrained-model for WSHP and then extract to './WSHP/parsing_network/models/'.
2. cd ./WSHP/parsing_network/
3. rm filename.txt
4. # config the dataset path in 'generate_flist.sh' (in line 3 - 7) and 'inference.py' (in line 29 - 42), details are shown in the following table.
5. chmod +x generate_flist.sh
6. sh generate_flist.sh
7. python inference.py
```

<table>
    <thead>
        <tr>
	    <th colspan="2">Configurations</th>
            <th>'hico train'</th>
            <th>'hico test'</th>
            <th>'vcoco train'</th>
            <th>'vcoco test'</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>generate_files.sh</td>
            <td>Line 4</td>
            <td>path=../../data/hico/ images/train2015/</td>
            <td>path=../../data/hico/ images/test2015/</td>
            <td>path=../../data/vcoco/ images/trainval/</td>
            <td>path=../../data/vcoco/ images/test/</td>
        </tr>
        <tr>
            <td rowspan=3>inference.py</td>
            <td>Line 29</td>
            <td colspan=2>dataset_name = 'hico'</td>
            <td colspan=2>dataset_name = 'vcoco'</td>
        </tr>
        <tr>
            <td>Line 30</td>
            <td>phrase = 'train'</td>
            <td>phrase = 'test'</td>
            <td>phrase = 'train'</td>
            <td>phrase = 'test'</td>
        </tr>
        <tr>
            <td>Line 39</td>
            <td colspan=2>DATASET_PATH = f"../../data/hico/images/{phrase+'2015'}/"</td>
            <td>DATASET_PATH = f"../../data/hico/ images/{phrase+'val'}/"</td>
            <td>DATASET_PATH = f"../../data/hico/ images/{phrase}/"</td>
        </tr>
    </tbody>
</table>

## Pretrained Model
You can download some pretrained models from:
- for [WSHP](https://doi.org/10.5281/zenodo.4506593): https://doi.org/10.5281/zenodo.4506593 or https://jbox.sjtu.edu.cn/l/hJjgjw
- for [hico](https://doi.org/10.5281/zenodo.4513884): https://doi.org/10.5281/zenodo.4513884
- for [vcoco](https://drive.google.com/file/d/1kBLOF3qj5SGEOZHh0HjCWtLSZGiPDlzJ/view?usp=sharing): https://drive.google.com/file/d/1kBLOF3qj5SGEOZHh0HjCWtLSZGiPDlzJ/view?usp=sharing

After that, you should extract them respectively in:
- for WSHP: './WSHP/parsing_network/models/'
- for hico: './weights/res101/hico_full/'
- for vcoco: './weights/res101/vcoco_full/'

Note: The pretrained model for vcoco provided here performs better than that in our paper, while we just increase some training epoches.

## Pre-computed Results
If you don't want to wait for test, we also provide our pre-computed results in test set for fast evaluation:
- for [hico](https://drive.google.com/file/d/1WRlXxVp-4cfnGCxfHEn1LGVrrA7kBJxn/view?usp=sharing): https://drive.google.com/file/d/1WRlXxVp-4cfnGCxfHEn1LGVrrA7kBJxn/view?usp=sharing
- for [vcoco](https://drive.google.com/file/d/1ShA8wCPEPBUGhCKM7Rxvk3wtfxBoF7ll/view?usp=sharing): https://drive.google.com/file/d/1ShA8wCPEPBUGhCKM7Rxvk3wtfxBoF7ll/view?usp=sharing

You can download and extract them to './output/hico_full/' and './output/vcoco_full/' respectively.

## Quick Start
#### Training
Before custom your own training progress, you have to download a pretrained weights to initialize the model([VGG16](https://doi.org/10.5281/zenodo.4515251) or [ResNet101](https://doi.org/10.5281/zenodo.4513878)), and put it into './data/pretrained_model/'.
You can train you own model by running the following script:  
```python
python trainval_net.py −−dataset hico_full −−checkepoch 6 --checkpoint 91451
```
or
```python
python trainval_net.py −−dataset vcoco_full −−checkepoch 18 --checkpoint 10051
```
#### Test
Based on the pretrained model, you can predict all the images in the dataset by running the following script:
```python
python test_net_hico.py
```
or
```python
python test_net_vcoco.py
```
If you have already downloaded the pre-computed test results, the detection stage will be automaticly skipped, and the test results will be evaluated.

#### Demo
After generating the test results, You can simply run scripts below to visualize the result of an random selected image.
```python
python demo_vcoco.py
```
You can also custom 'im_id' param to select a specific image to visualize, or 'show_category' param to select whether to show interaction and object categories or not.
The complete script is as follows:
```python
python demo_vcoco.py --im_id <int> --show_category <True | False>
```

## Link
- We construct our code based on [Faster-RCNN](https://github.com/jwyang/faster-rcnn.pytorch)(https://github.com/jwyang/faster-rcnn.pytorch), and follow [TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network)(https://github.com/jwyang/faster-rcnn.pytorch) to build our evaluation code.
- We use [WSHP](https://github.com/MVIG-SJTU/WSHP)(https://github.com/MVIG-SJTU/WSHP) to parse human body part regions, and use its pre-trained weights [here](https://doi.org/10.5281/zenodo.4506593)(https://doi.org/10.5281/zenodo.4506593 or https://jbox.sjtu.edu.cn/l/hJjgjw).
- We provide the two datasets we used in our experiments as:
[hico](https://drive.google.com/file/d/1KFXWUt6lCvXGflpq6tvrfqtPES6VoPbk/view?usp=sharing): https://drive.google.com/file/d/1KFXWUt6lCvXGflpq6tvrfqtPES6VoPbk/view?usp=sharing
[vcoco](https://drive.google.com/file/d/1NlyrqhtYUlQCkLXJzM036AqCeo8aVjs9/view?usp=sharing): https://drive.google.com/file/d/1NlyrqhtYUlQCkLXJzM036AqCeo8aVjs9/view?usp=sharing
- We provide our pre-trained weights as:
[for hico](https://doi.org/10.5281/zenodo.4513884): https://doi.org/10.5281/zenodo.4513884
[for vcoco](https://drive.google.com/file/d/1kBLOF3qj5SGEOZHh0HjCWtLSZGiPDlzJ/view?usp=sharing): https://drive.google.com/file/d/1kBLOF3qj5SGEOZHh0HjCWtLSZGiPDlzJ/view?usp=sharing
- We provide our pre-computed results for fast evaluation as:
[for hico](https://drive.google.com/file/d/1WRlXxVp-4cfnGCxfHEn1LGVrrA7kBJxn/view?usp=sharing): https://drive.google.com/file/d/1WRlXxVp-4cfnGCxfHEn1LGVrrA7kBJxn/view?usp=sharing
[for vcoco](https://drive.google.com/file/d/1ShA8wCPEPBUGhCKM7Rxvk3wtfxBoF7ll/view?usp=sharing): https://drive.google.com/file/d/1ShA8wCPEPBUGhCKM7Rxvk3wtfxBoF7ll/view?usp=sharing
- If you want to custom the training, you can achieve the pretrained model for two avalable backbones:

VGG16: 
<br>
	[Dropbox](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0): https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0
<br>
	[VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth): https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/vgg16_caffe.pth
<br>
	[Zenodo](https://doi.org/10.5281/zenodo.4515251): https://doi.org/10.5281/zenodo.4515251
<br>
ResNet101: 
<br>
	[Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0): https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0
<br>
	[VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth): https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth
<br>
	[Zenodo](https://doi.org/10.5281/zenodo.4513878): https://doi.org/10.5281/zenodo.4513878

Collection of data resources above: https://zenodo.org/record/5823557

## Citation

    @inproceedings{inproceedings,
		author = {Sun, Xu and Hu, Xinwen and Ren, Tongwei and Wu, Gangshan},
		year = {2020},
		month = {06},
		pages = {26-34},
		title = {Human Object Interaction Detection via Multi-level Conditioned Network},
		doi = {10.1145/3372278.3390671}
	}
