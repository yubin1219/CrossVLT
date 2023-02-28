# Cross-aware Early Fusion with Stage-divided Vision and Language Transformer Encoders for Referring Image Segmentation

<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/201589583-0bedfbc7-e28a-4f78-9135-f37612c3c7d8.png" width="90%" height="90%">
</p>
<br/>
This repo is the implementation of "Cross-aware Early Fusion with Stage-divided Vision and Language Transformer
Encoders for Referring Image Segmentation" and is organized as follows: 

* `./train.py` is implemented to train the model.
* `./test.py` is implemented to evaluate the model.
* `./refer` contains data pre-processing manual and code.
* `./data/dataset_refer_bert.py` is where the dataset class is defined.
* `./lib` contains codes implementing vision encoder and segmentation decoder.
* `./bert` contains codes migrated from Hugging Face, which implement the BERT model. We modified some codes to implement our stage-divided language encoder.
* `./CrossVLT.py` is implemented for the main network, which consists of the stage-divided vision and language encoders and simple segmentation decoder.
* `./utils.py` defines functions that track training statistics and setup functions for `Distributed DataParallel`.


## Installation and Setup
### **Environment**

This repo requires Pytorch v 1.7.1 and Python 3.8.
Install Pytorch v 1.7.1 with a CUDA version that works on your cluster. We used CUDA 11.0 in this repo:
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
Then, install the packages in `requirements.txt` via pip:
```
pip3 install -r requirements.txt
```
### **Initialization with Pretrained weights for Training**

Create the `./pretrained` directory.
```
mkdir ./pretrained
```
Download [ImageNet pretrained weights of the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and [official weights of the BERT (Pytorch, uncased version)](https://huggingface.co/google/bert_uncased_L-12_H-768_A-12/blob/main/pytorch_model.bin) into the `./pretrained` folder.

### **Trained weights of CrossVLT for Testing**

Create the `./checkpoints` directory.
```
mkdir ./checkpoints
```
Download [CrossVLT model weights](https://drive.google.com/drive/folders/1zEAuhntMjH13rhzSYibw_xXLvp-4xB_v?usp=sharing) into the `./checkpoints` folder.

### **Datasets**

Follow [README.md](https://anonymous.4open.science/r/CrossVLT-E384/refer/README.md) in the `./refer` directory to set up subdirectories and download annotations.
Download 2014 Train images [83K/13GB] from [COCO](https://cocodataset.org/#download), and extract the downloaded `train_2014.zip` file to `./refer/data/images/mscoco/images`. 

## Training

We use `DistributedDataParallel` from PyTorch. The CrossVLT were trained using 2 x 24G RTX3090 cards.
To run on multi GPUs (2 GPUs is used in this example) on a single node:
```
mkdir ./models

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 train.py --dataset refcoco --swin_type base --lr 0.00003 --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 train.py --dataset refcoco+ --swin_type base --lr 0.00003 --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco+

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 train.py --dataset refcocog --splitBy umd --swin_type base --lr 0.00003 --epochs 40 --img_size 480 2>&1 | tee ./models/refcocog
```
To store the training logs, we need to manually create the `./models` directory via `mkdir` before running `train.py`.
* *--dataset* is the dataset name. One can choose from `refcoco`, `refcoco+`, and `refcocog`.
* *--splitBy* needs to be specified if and only if the dataset is G-Ref (which is also called RefCOCOg).
* *--swin_type* specifies the version of the Swin Transformer. One can choose from `tiny`, `small`, `base`, and `large`. The default is `base`.

## Testing

To evaluate, run one of:
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --swin_type base --dataset refcoco --split val --resume ./checkpoints/best_refcoco.pth --workers 1 --img_size 480

CUDA_VISIBLE_DEVICES=0 python3 test.py --swin_type base --dataset refcoco+ --split val --resume ./checkpoints/best_refcoco+.pth --workers 1 --img_size 480

CUDA_VISIBLE_DEVICES=0 python3 test.py --swin_type base --dataset refcocog --splitBy umd --split val --resume ./checkpoints/best_refcocog.pth --workers 1 --img_size 480
```
* *--split* is the subset to evaluate. One can choose from `val`, `testA`, and `testB` for RefCOCO/RefCOCO+, and `val` and `test` for G-Ref (RefCOCOg).
* *--resume* is the path to the weights of a trained CrossVLT.

## Results
The complete evaluation results of the proposed model are summarized as follows:

|     Dataset     | P@0.5 | P@0.6 | P@0.7 | P@0.8 | P@0.9 | Mean IoU | Overall IoU |
|:---------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:--------:|:-----------:|
| RefCOCO val     | 85.82 | 82.80 | 77.49 | 67.21 | 35.97 |    75.48 |   73.44     |
| RefCOCO test A  | 88.92 | 86.25 | 81.62 | 69.79 | 36.68 |  77.54   |    76.16    |
| RefCOCO test B  | 81.35 | 77.74 | 72.27 | 62.63 | 37.64 |  72.69   |    70.15    |
| RefCOCO+ val    | 76.41 | 73.09 | 68.49 | 59.09 | 31.39 |    67.27 |   63.60     |
| RefCOCO+ test A | 82.43 | 79.57 | 74.80 | 64.65 | 32.89 |    72.00 |   69.10     |
| RefCOCO+ test B | 67.19 | 63.31 | 58.89 | 49.93 | 29.15 |  60.09   |    55.23    |
| G-Ref val       | 74.75 | 70.45 | 64.58 | 54.21 | 28.57 |    66.21 |   62.68     |
| G-Ref test      | 71.54 | 66.38 | 59.00 | 48.21 | 23.10 |    62.09 |   63.75     |

## Visualization
Comparison with ablated models (without feature-based alignment / without cross-aware early fusion / without both components (Basic model)) :
<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/221766582-cbb3c4b2-892b-48c7-aafb-3b9caaaf9c88.png" width="80%" height="80%">
</p>
<br/>

Comparison with previous state-of-the-art models (late fusion model (VLT) / vision-only early fusion model (LAVT)) :
<p align="center">
    <img src="https://user-images.githubusercontent.com/74402562/202836985-3c4ce113-cdd6-4283-9be4-545a7d732a80.png" width="90%" height="90%">
</p>
<br/>
