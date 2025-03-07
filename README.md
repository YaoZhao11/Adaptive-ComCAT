# Adaptive-ComCAT

ComCAT is a low-rank decomposition method for VisionTransformer, using NAS searching for rank-values. The model are available at https://doi.org/10.48550/arXiv.2305.17235

For Adaptive-ComCAT, introducing an adaptive coefficient for searching frequency to reducing unnecessary NAS iterations so as to alleviate the impairment of huge searching space.

The input image is segmented into patches of the sequence and sent to the transformer encoder by position coding. Each block has 1 attention layer, corresponding to two weight matrices WQK and WVO, and 2 full connection layers, corresponding to W1 and W2. After SVD low-rank decomposition, the original weight matrix is replaced. After multi layer perceptron (MLP), the function of image classification can be realized. By monitoring the difference of Rank matrix, the frequency coefficient in the epoch t-1 iteration, i.e. F(t-1), is transferred to the epoch t, i.e. F(t), stage through the dynamic programming method, so as to reduce the use frequency of neural architecture search(NAS) and accelerate the effect of model training.

![image](https://github.com/user-attachments/assets/f76eef9c-f4a1-4920-b428-704dcf0bd439)


# Dependencies

GPU NVIDIA RTX A6000 48GB

Ubuntu 20.04

Python 3.9

Pytorch 2.0.1

CUDA 11.07

timm 0.5.4

# Dataset
In this paper, we use ImageNet dataset, it is a public available dataset, and from the reference below:

Deng J, Dong W, Socher R et al (2009) Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, IEEE.pp 248-255.

we can also downlond dataset in https://image-net.org/challenges/LSVRC/2012/index.php

# Setup
```
conda install -c pytorch pytorch torchvision
pip install timm==0.5.4
```
# Pre-trained model
```
# DeiT-small
wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
# DeiT-Base
wget https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
```

# Training
```
# DeiT-small
python main.py --model deit_small_patch16_224 --data-path /path/to/imagenet/ --batch-size 512 --load deit_small_patch16_224-cd65a155.pth  --output_dir small_auto  --epochs 30 --warmup-epochs 0 --search-rank --distillation-type hard --teacher-model deit_small_patch16_224 --teacher-path  https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth --with-align --distillation-without-token --batch-size-search 64 --target-params-reduction 0.5 > small_auto.log

# DeiT-Base
python main.py --model deit_base_patch16_224 --data-path /path/to/imagenet/ --batch-size 320 --load deit_base_patch16_224-b5f2ef4d.pth --output_dir base_auto  --epochs 30 --warmup-epochs 0 --search-rank --distillation-type hard --teacher-model deit_base_patch16_224 --teacher-path  https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth --with-align --distillation-without-token --batch-size-search 16 --target-params-reduction 0.6 > base_auto.log
```

# Inference
```
# DeiT-small
mkdir small_79.58_0.44
python -m torch.distributed.launch --nproc_per_node=4 --use_env  main.py --model deit_small_patch16_224 --data-path /path/to/imagenet/ --batch-size 256 --finetune-rank-dir small_79.58_0.44 --attn2-with-bias --eval
# DeiT-Base
mkdir base_82.28_0.61
python -m torch.distributed.launch --nproc_per_node=4 --use_env  main.py --model deit_base_patch16_224 --data-path /path/to/imagenet/ --batch-size 256 --finetune-rank-dir base_82.28_0.61 --attn2-with-bias --eval
```
# Title of Article and Journal
Optimizing Low-Rank Decomposition for Efficient Attention-Based Vision Models via Adaptive Neural Architecture Search

《The Visual Computer》
