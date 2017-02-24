# PyramidNet_with_Stochastic_Depth

This repository contains the code for the paper "Deep Pyramidal Residual Networks with Separated Stochastic Depth" (https://arxiv.org/abs/1612.01230). 

The code is based on Facebook's implementation of ResNet (https://github.com/facebook/fb.resnet.torch), PyramidNet (https://github.com/jhkim89/PyramidNet) and fb.resnet.torch-lesion-study (https://github.com/gcr/fb.resnet.torch-lesion-study).

## Usage

0. Install Torch (http://torch.ch) and ResNet (https://github.com/facebook/fb.resnet.torch).
1. Add the files pyramiddrop.lua, pyramidsepdrop.lua and StochasticDrop.lua (https://github.com/gcr/fb.resnet.torch-lesion-study/tree/master/models) to the folder "models".
2. Change the learning rate schedule in the file train.lua: "decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0" to "decay = epoch >= 225 and 2 or epoch >= 150 and 1 or 0".
3. Train our Networks, by running main.lua as below:

To train PyramidDrop-110 (alpha=90) on CIFAR-10 dataset:
```bash
th main.lua -dataset cifar10 -nEpochs 300 -LR 0.5 -netType pyramiddrop -batchSize 128 -shareGradInput true
```

To train PyramidDrop-110 (alpha=90) with 4 GPUs on CIFAR-10 dataset:

Change the code in the file models/init.lua

```bash
:threads(function()
  local cudnn = require 'cudnn'
  cudnn.fastest, cudnn.benchmark = fastest, benchmark
end)
```

to

```bash
Change the code in the file models/init.lua
:threads(function()
  local cudnn = require 'cudnn'
  require 'models/StochasticDrop'
  cudnn.fastest, cudnn.benchmark = fastest, benchmark
end)
```

and

```bash
th main.lua -dataset cifar10 -nEpochs 300 -LR 0.5 -netType pyramiddrop -batchSize 128 -shareGradInput true -nGPU 4 -nThreads 8
```
