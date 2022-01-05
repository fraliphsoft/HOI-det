## Instance of Interest Detection

This branch <b>'pytorch1.x'</b> is a renewed version for the support of python 3.x and pytorch 1.x.

We validate this branch with the following environment:

(1) Ubuntu 16.04
(2) CPU Intel(R) Xeon(R) E5-2680 v4
(3) GPU GeForce RTX 3090 @ 24GB
(4) CUDA 11.1 and cuDNN 8.0.5.
(5) Python 3.8.12, torch=1.9.0, tensorflow=2.6.0.
(6) Matlab R2021b.

The installation process of this branch differs from that of 'master' branch (with python 2.7 and pytorch 0.4.0), and other scripts are consistant.

## Requirements

certifi==2021.10.8
cffi==1.14.6
cycler==0.11.0
Cython==0.29.26
easydict==1.9
kiwisolver==1.3.2
matplotlib==3.5.1
numpy==1.19.2
opencv-python==4.5.5.62
Pillow==8.4.0
protobuf==3.19.1
pycparser==2.21
pyparsing==3.0.6
python-dateutil==2.8.2
PyYAML==6.0
scipy==1.6.2
six==1.15.0
tensorboardX==2.4.1
tensorflow-gpu==2.6.0
torch==1.9.0+cu111
torchvision==0.10.0+cu111
tqdm==4.62.3
tf_slim==1.1.0
imageio==2.13.5

## Installation

#### Install Environment

```
#---optional---
conda create --name <env-name> python=3.8
conda activate <env-name>
```

```
pip install -r requirements.txt
# for cuda environment problems, you maybe need to install spercific version of tensorflow or torch, using the command below
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
# conda install tensorflow-gpu==2.6.0

```

#### Install Matlab

We adopt the implementation of evaluation functions in HICO dataset from [TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network)(actually from [iCAN](https://github.com/vt-vl-lab/iCAN) or former), so matlab runtime environment is needed.
There are no restrictions on versions, you can download matlab from its official website and install according to the tutorial.

#### Install Libraries

```
1. cd ./lib/
2. pip install -e .
# then there should be some file like
#   /lib/pycocotools/_mask.cpython-38-x86_64-linux-gnu.so
#   /lib/model/_C.cpython-38-x86_64-linux-gnu.so
#   /lib/model/utils/cython_bbox.cpython-38-x86_64-linux-gnu.so
```

Note that the compilation process of 'master' branch and this 'pytorch1.x' branch is totally different, thus it is not recommended that the two branches share a same local directory path.
In other words, after the installation scipts 'sh make.sh' or 'pip install' are done, the command 'git checkout' is not supported.

## Prepare Data

#### 1. Download Datasets

We perform our experiments in 'vcoco' and 'hico(v2016)' dataset. We follow the setting of [TIN](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network) as our original dataset, and then use [WSHP](https://github.com/MVIG-SJTU/WSHP) to extract human body part regions in hico and vcoco dataset respectively.
Considering that the body part regions data is too large(>100GB for test and >500GB for training) while the code runs fast, we provide prepared dataset except human body part regions:

- [vcoco](https://drive.google.com/file/d/1NlyrqhtYUlQCkLXJzM036AqCeo8aVjs9/view?usp=sharing): https://drive.google.com/file/d/1NlyrqhtYUlQCkLXJzM036AqCeo8aVjs9/view?usp=sharing
- [hico](https://drive.google.com/file/d/1KFXWUt6lCvXGflpq6tvrfqtPES6VoPbk/view?usp=sharing): https://drive.google.com/file/d/1KFXWUt6lCvXGflpq6tvrfqtPES6VoPbk/view?usp=sharing

You can download the datasets and extract them to './data/hico/' and './data/vcoco/' respectively.

then run the commands below to generate the required hico data file

```shell
cd data/hico
cp anno.mat anno_full.mat
cp anno_bbox.mat anno_bbox_full.mat
```

note that there are dirty data in hico datasets with incorrect rotated images, which lead to failure when fusing the image and some preprocessed data because of different tensor shape. So we pick up these image and rotate them. For train set, see lines 77-86 and 117-120  in `/lib/roi_data_layer/minibatch.py`, and lines 70-77 in `/lib/roi_data_layer/roibatchLoader.py`. For test set, see lines 282-295, 307-310 and 316-319 in `/test_net_hico.py`.

#### 2. Calculate Human Body Part Regions

We provide the fine-tuned [WSHP](https://github.com/MVIG-SJTU/WSHP) code in './WSHP', you can run it following these steps:

```
1. # download pretrained-model for WSHP and then extract to './WSHP/parsing_network/models/'.
2. cd ./WSHP/parsing_network/
3. rm filename.txt
4. # config the dataset path in 'generate_flist.sh' (in line 3 - 7)
5. chmod +x generate_flist.sh
6. sh generate_flist.sh
7. # config the dataset path in 'inference.py' (in line 29 - 42)
8. python inference.py
9. # if using the default path, move the output from './WSHP/parsing_network/output/' to './data/hico/humans/' or './data/vcoco/humans/'
```

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
python trainval_net.py −−dataset hico_full  --epochs 6 --lr_decay_step 1
```

or

```python
python trainval_net.py −−dataset vcoco_full  --epochs 18 --lr_decay_step 3
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

## Citation

    @inproceedings{inproceedings,
    	author = {Sun, Xu and Hu, Xinwen and Ren, Tongwei and Wu, Gangshan},
    	year = {2020},
    	month = {06},
    	pages = {26-34},
    	title = {Human Object Interaction Detection via Multi-level Conditioned Network},
    	doi = {10.1145/3372278.3390671}
    }