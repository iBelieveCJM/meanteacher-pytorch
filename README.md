# MeanTeacher-PyTorch
The repository implement a semi-supervised deep learning method, MeanTeacher. More details for the method please refer to *Mean teacher are better role models: Wegiht-averaged consistency targets improve semi-supervised deep learning results*.

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
python -m experiments.mt_test
```

## Visualization:

Make sure you have installed the tensorflow for tensorboard
```shell
tensorboard --logdir runs
```

## Code Reference

[mean-teacher@CuriousAI](https://github.com/CuriousAI/mean-teacher)
