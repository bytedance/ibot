# Installation

Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. This repository should be used with [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), [mmsegmentation==0.12.0](https://github.com/open-mmlab/mmsegmentation/releases/tag/v0.12.0), and [cyanure](https://github.com/jmairal/cyanure) for evaluation on downstream tasks. To get the full dependencies, please run:

```
pip3 install -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html mmcv-full==1.3.9
pip3 install pytest-runner scipy tensorboardX faiss-gpu==1.6.1 tqdm lmdb sklearn pyarrow==2.0.0 timm DALL-E munkres six einops

# install apex
pip3 install git+https://github.com/NVIDIA/apex \
    --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"

# install mmdetection for object detection & instance segmentation
git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
cd Swin-Transformer-Object-Detection
pip3 install -r requirements/build.txt
pip3 install -v -e .
cd ..

# install mmsegmentation==0.12.0 for semantic segmentation
git clone -b v0.12.0 https://github.com/open-mmlab/mmsegmentation
cd mmsegmentation
pip3 install -v -e .
cd ..

# install cyanure-mkl for logistic regression
pip3 install mkl
git clone https://github.com/jmairal/cyanure.git
cd cyanure
sudo python3 setup_cyanure_mkl.py install
cd ..
```
