# MeanTeacher-PyTorch
The repository implement two semi-supervised deep learning methods, MeanTeacher and PI model. More details for the method please refer to *Mean teacher are better role models: Wegiht-averaged consistency targets improve semi-supervised deep learning results* and *Temporal Ensembling for Semi-supervised Learning*.

This repository is based on the official repository of [mean-teacher@CuriousAI](https://github.com/CuriousAI/mean-teacher). And there are not only ConvLarge Net implemented, but also other popular networks (come from [pytorch-cifar@kuangliu](https://github.com/kuangliu/pytorch-cifar)).

I implemet PI model as a special case of MeanTeacher when smoothing coefficient hyperparameter is zero.

## The environment:

- Python 3.6.5::Anaconda

- PyTorch >= 0.4.0

- torchvision 0.2.1

- tensorboardX (for log)

- tensorflow (for visualization)

## To prepare the data:
```shell
bash data-local/bin/prepare_cifar10.sh
```

## To run the code:
```shell
python -m experiments.ema_test
```
or
```shell
python -m experiments.pi_test
```

## Visualization:

Make sure you have installed the tensorflow for tensorboard
```shell
tensorboard --logdir runs
```

## Code Reference

[pytorch-cifar@kuangliu](https://github.com/kuangliu/pytorch-cifar)

[mean-teacher@CuriousAI](https://github.com/CuriousAI/mean-teacher)
