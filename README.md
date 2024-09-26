# Generalizing Consistency Policy to Visual RL with Prioritized Proximal Experience Regularization

<p align="center" style="font-size: 50px">
   <a href="https://arxiv.org/pdf/<ARXIV PAPER ID>">[Paper]</a>&emsp;<a href="https://jzndd.github.io/CP3ER-Page/">[Project Website]</a>
</p>

## Overview
This is the official PyTorch implementation of the paper "Generalizing Consistency Policy to Visual RL with Prioritized Proximal Experience Regularization". Our approach, CP3ER, significantly enhances the stability and performance of visual reinforcement learning models. 

## Installation

### Setup
To install the required packages for DeepMind Control Suite and Metaworld, please run the following commands:
```bash
conda env create -f cp3er.yaml # for dmc
 
conda env activate -f cp3ermw.yaml  # for metaworld
```

Then, install the Metaworld package:
```bash
conda activate cp3ermw
cd Metaworld
pip install -e .
```

## Reproducing Experimental Results
### Training for dmc tasks
```bash
python train.py task_name=acrobot_swingup
```
You can decide whether to use wandb to log your experiment process by specifying the 'use_wb' parameter, and determine whether to use a GPU for training by specifying the 'device' parameter. For more parameter options, please refer to the cfgs/config.yaml file.
```bash
python train.py task_name=cheetah_run device=cuda:1 use_wb=True seed=1
```

### Training for metaworld tasks
Similar to training for DMC tasks, you can run the following scripts for testing in Metaworld:
```bash
python train_mw.py task_name=assembly-v2
```

## Acknowledgement
CP3ER is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank DrQ-v2 authors for open-sourcing the DrQv2 codebase. Our implementation builds on top of their repository.