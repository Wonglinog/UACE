# Uncertainty-Aware Cross Entropy for Robust Learning with Noisy Labels
This repository is the official implementation of [Uncertainty-Aware Cross Entropy for Robust Learning with Noisy Labels].

## Requirements
Python >= 3.7, torch >= 1.7, torchvision >= 0.4.1, hydra-core >= 1.1.0, numpy >= 1.11.2, black == 20.8b1

## Training

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python3 train.py --multirun dataset=mnist dataset.train.asym=0,1,2 model=layer_4 loss=ce,uace,gce,sce,tce dataset.train.corrupt_prob=0.4 dataset.train.noise_seed=0,1,2 mixed_precision=true

## Motivation

## Methods

## Experiments

### Results (paired t-test at 0.05 significance level)
doc/Table1.png
### Representations

### Ablation Study 

### Parametric Analysis

