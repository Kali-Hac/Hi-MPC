![Python >=3.5](https://img.shields.io/badge/Python->=3.5-blue.svg)
![Tensorflow >=1.14.0](https://img.shields.io/badge/Tensorflow->=1.14.0-yellow.svg)
![Pytorch >=1.1.0](https://img.shields.io/badge/Pytorch->=1.1.0-green.svg)
![Faiss-gpu >= 1.6.3](https://img.shields.io/badge/Faiss->=1.6.3-orange.svg)

# Hierarchical Skeleton Meta-Prototype Contrastive Learning with Hard Skeleton Mining for Unsupervised Person Re-Identification
By Haocong Rao, Cyril Leung, and Chunyan Miao. In *International Journal of Computer Vision (IJCV), 2023 (In Press)* ([**Arxiv**](https://arxiv.org/abs/2307.12917)), ([**Paper**](https://link.springer.com/article/10.1007/s11263-023-01864-0)).

## Introduction
This is the official implementation of Hi-MPC model presented by "Hierarchical Skeleton Meta-Prototype Contrastive Learning with Hard Skeleton Mining for Unsupervised Person Re-Identification". The codes are used to reproduce experimental results of the proposed Multi-level Skeleton Meta-Representation (MSMR) in the paper.

![image](https://github.com/Kali-Hac/Hi-MPC/blob/main/img/overview.jpg)
Abstract: With rapid advancements in depth sensors and deep learning, skeleton-based person re-identification (re-ID) models have recently achieved remarkable progress with many advantages. Most existing solutions learn single-level skeleton features from body joints with the assumption of equal skeleton importance, while they typically lack the ability to exploit more informative skeleton features from various levels such as limb level with more global body patterns. The label dependency of these methods also limits their flexibility in learning more general skeleton representations. This paper proposes a generic unsupervised Hierarchical skeleton Meta-Prototype Contrastive learning (Hi-MPC) approach with Hard Skeleton Mining (HSM) for person re-ID with unlabeled 3D skeletons. Firstly, we construct hierarchical representations of skeletons to model coarse-to-fine body and motion features from the levels of body joints, components, and limbs. Then a hierarchical meta-prototype contrastive learning model is proposed to cluster and contrast the most typical skeleton features ("prototypes") from different-level skeletons. By converting original prototypes into meta-prototypes with multiple homogeneous transformations, we induce the model to learn the inherent consistency of prototypes to capture more effective skeleton features for person re-ID. Furthermore, we devise a hard skeleton mining mechanism to adaptively infer the informative importance of each skeleton, so as to focus on harder skeletons to learn more discriminative skeleton representations. Extensive evaluations on five datasets demonstrate that our approach outperforms a wide variety of state-of-the-art skeleton-based methods. We further show the general applicability of our method to cross-view person re-ID and RGB-based scenarios with estimated skeletons.

## Environment
- Python >= 3.5
- Tensorflow-gpu >= 1.14.0
- Pytorch >= 1.1.0
- Faiss-gpu >= 1.6.3

Here we provide a configuration file to install the extra requirements (if needed):
```bash
conda install --file requirements.txt
```

**Note**: This file will not install tensorflow/tensorflow-gpu, faiss-gpu, pytroch/torch, please install them according to the cuda version of your graphic cards: [**Tensorflow**](https://www.tensorflow.org/install/pip), [**Pytorch**](https://pytorch.org/get-started/locally/). Take cuda 9.0 for example:
```bash
conda install faiss-gpu cuda90 -c pytorch
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
conda install tensorflow-gpu==1.14
conda install scikit-learn
```

## Datasets and Models
We provide three already pre-processed datasets (IAS-Lab, BIWI, KGBD) with various sequence lengths (**f=4/6/8/10/12**) [**here (pwd: 7je2)**](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg) and the **pre-trained models** [**here (pwd: xqho)**](https://pan.baidu.com/s/19sNRr3z71ufGPjQF9pvsMw). Since we report the average performance of our approach on all datasets, here the provided models may produce better results than the paper. <br/>

Please download the pre-processed datasets and model files while unzipping them to ``Datasets/`` and ``ReID_Models/`` folders in the current directory. <br/>

**Note**: The access to the Vislab Multi-view KS20 dataset and large-scale RGB-based gait dataset CASIA-B are available upon request. If you have signed the license agreement and been granted the right to use them, please email us with the signed agreement and we will share the complete pre-processed KS20 and CASIA-B data. The original datasets can be downloaded here: [IAS-Lab](http://robotics.dei.unipd.it/reid/index.php/downloads), [BIWI](http://robotics.dei.unipd.it/reid/index.php/downloads), [KGBD](https://www.researchgate.net/publication/275023745_Kinect_Gait_Biometry_Dataset_-_data_from_164_individuals_walking_in_front_of_a_X-Box_360_Kinect_Sensor), [KS20](http://vislab.isr.ist.utl.pt/datasets/#ks20), [CASIA-B](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp). We also provide the ``Preprocess.py`` for directly transforming original datasets to the formated training and testing data. <br/> 

## Dataset Pre-Processing
To (1) extract 3D skeleton sequences of length **f=6** from original datasets and (2) process them in a unified format (``.npy``) for the model inputs, please simply run the following command: 
```bash
python Preprocess.py 6
```
**Note**: If you hope to preprocess manually (or *you can get the [already preprocessed data (pwd: 7je2)](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg)*), please frist download and unzip the original datasets to the current directory with following folder structure:
```bash
[Current Directory]
├─ BIWI
│    ├─ Testing
│    │    ├─ Still
│    │    └─ Walking
│    └─ Training
├─ IAS
│    ├─ TestingA
│    ├─ TestingB
│    └─ Training
├─ KGBD
│    └─ kinect gait raw dataset
└─ KS20
     ├─ frontal
     ├─ left_diagonal
     ├─ left_lateral
     ├─ right_diagonal
     └─ right_lateral
```
After dataset preprocessing, the auto-generated folder structure of datasets is as follows:
```bash
Datasets
├─ BIWI
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ Still
│      │    └─ Walking
│      └─ train_npy_data
├─ IAS
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ A
│      │    └─ B
│      └─ train_npy_data
├─ KGBD
│    └─ 6
│      ├─ test_npy_data
│      │    ├─ gallery
│      │    └─ probe
│      └─ train_npy_data
└─ KS20
    └─ 6
      ├─ test_npy_data
      │    ├─ gallery
      │    └─ probe
      └─ train_npy_data
```
**Note**: KS20 data need first transforming ".mat" to ".txt". If you are interested in the complete preprocessing of KS20 and CASIA-B, please contact us and we will share. We recommend to directly download the preprocessed data [**here (pwd: 7je2)**](https://pan.baidu.com/s/1R7CEsyMJsEnZGFLqwvchBg).

## Model Usage

To (1) train the unsupervised Hi-MPC model to obtain skeleton representations (MSMR) and (2) validate their effectiveness on the person re-ID task on a specific dataset (probe), please simply run the following command:  

```bash
python Hi-MPC.py --dataset KS20 --probe probe

# Default options: --dataset KS20 --probe probe --length 6  --gpu 0
# --dataset [IAS, KS20, BIWI, KGBD]
# --probe ['probe' (the only probe for KS20 or KGBD), 'A' (for IAS-A probe), 'B' (for IAS-B probe), 'Walking' (for BIWI-Walking probe), 'Still' (for BIWI-Still probe)] 
# --length [4, 6, 8, 10, 12] 
# --(H, M, eps, min_samples, lr, etc.) with default settings for each dataset
# --mode [Train (for training), Eval (for testing)]
# --gpu [0, 1, ...]

```

To print Rank-1 accuracy, Rank-5 accuracy, Rank-10 accuracy and mAP when applying different level representations (joint-level, component-level, limb-level, MSMR) of the best model saved in default directory (```ReID_Models/(Dataset)/(Probe)```), run:

```bash
python Hi-MPC.py --dataset KS20 --probe probe --mode Eval
```

## Application to Model-Estimated Skeleton Data 

### Estimate 3D Skeletons from RGB-Based Scenes
To apply our Hi-MPC to person re-ID under the large-scale RGB scenes (CASIA B), we exploit pose estimation methods to extract 3D skeletons from RGB videos of CASIA B as follows:
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Extract the 2D human body joints by using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Step 3: Estimate the 3D human body joints by using [3DHumanPose](https://github.com/flyawaychase/3DHumanPose)


We provide already pre-processed skeleton data of CASIA B for **single-condition** (Nm-Nm, Cl-Cl, Bg-Bg) and **cross-condition evaluation** (Cl-Nm, Bg-Nm) (**f=40/50/60**) [**here (pwd: 07id)**](https://pan.baidu.com/s/1_Licrunki68r7F3EWQwYng). 
Please download the pre-processed datasets into the directory ``Datasets/``. <br/>

### Usage
To (1) train the Hi-MPC model to obtain skeleton representations (MSMR) and (2) validate their effectiveness on the person re-ID task on CASIA B under **single-condition** and **cross-condition** settings, please simply run the following command:

```bash
python Hi-MPC.py --dataset CASIA_B --probe_type nm.nm --length 40

# --length [40, 50, 60] 
# --probe_type ['nm.nm' (for 'Nm' probe and 'Nm' gallery), 'cl.cl', 'bg.bg', 'cl.nm' (for 'Cl' probe and 'Nm' gallery), 'bg.nm']  
# --(H, M, eps, min_samples, lr, etc.) with default settings
# --gpu [0, 1, ...]

```

Please see ```Hi-MPC.py``` for more details.

## Citation
If you found this repository useful, please consider citing:
```bash
@article{rao2023hierarchical,
  title={Hierarchical Skeleton Meta-Prototype Contrastive Learning with Hard Skeleton Mining for Unsupervised Person Re-Identification},
  author={Rao, Haocong and Leung, Cyril and Miao, Chunyan},
  journal={arXiv preprint arXiv:2307.12917},
  year={2023}
}
```

## License

Hi-MPC is released under the MIT License. Our models and codes must only be used for legitimate research.
