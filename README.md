# Multi-network Ensembling for GAN Training and Adversarial Attacks

*Shuting Zheng,Yuan-Gen Wang*

# Abstract

Deep neural networks are fragile to attacks from adversarial examples. However, successfully fooling a target model with a limited query budget is challenging in black-box scenarios where none of the network architecture, parameters, or training data is available. An alternative solution is to employ a generative adversarial network (GAN) to generate synthetic data and train a substitute model of the target model, allowing us to perform the white-box attack, namely the transfer attack. We find that the current single network substitution suffers from a performance bottleneck. This article presents a multi-network ensembling to optimize GAN training and adversarial example generation, which includes a new multi-network substitute training strategy and an adaptive ensemble attack strategy. Extensive experiments on the MNIST and CIFAR-10 datasets show that our method outperforms the state-of-the-art in terms of query efficiency. Especially, when attacking the Microsoft Azure online model in both the label-only and probability-only scenarios, our method achieves a 100% attack success rate with a meager query budget.

# Prerequisites

- Python >= 3.7
- PyTorch >= 1.13
- NVIDIA GPU + CUDA cuDNN

# Getting Started

## Installation

- Clone this repo:

```
https://github.com/GZHU-DVL/ZHENG.git
cd ZHENG
```

- Install PyTorch and dependencies from [http://pytorch.org](http://pytorch.org/)
- Install python requirements:

```
pip install -r requirements.txt
```

## Training

If you want to train the substitutes in MNIST:

```
python main_mnist.py  
```

If you want to train the substitutes in CIFAR10:

```
python main_cifar10.py
```

## Attacking

- Untargeted attack：

```
python evaluation_mnist.py --adv FGSM 
```
```
python evaluation_cifar10.py --adv FGSM 
```
- Targeted attack：

```
python evaluation_mnist.py --adv FGSM --target
```
```
python evaluation_cifar10.py --adv FGSM --target
```
# Acknowledge

The code is developed based on DaST: https://github.com/zhoumingyi/DaST
