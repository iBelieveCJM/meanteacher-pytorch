#!coding:utf-8
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from util import datasets, consTrainer
from util.datasets import TransformTwice as twice
from util.ramps import exp_warmup
from architectures.arch import arch

from util.datasets import NO_LABEL

def create_data_loaders(train_transform, 
                        eval_transform, 
                        datadir,
                        config):
    if config.twice:
        train_transform = twice(train_transform)
        eval_transform = twice(eval_transform)
    traindir = os.path.join(datadir, config.train_subdir)
    trainset = torchvision.datasets.ImageFolder(traindir, train_transform)
    if config.labels:
        with open(config.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = datasets.relabel_dataset(trainset, labels)
    assert len(trainset.imgs) == len(labeled_idxs)+len(unlabeled_idxs)
    if config.labeled_batch_size < config.batch_size:
        assert len(unlabeled_idxs)>0
        batch_sampler = datasets.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, config.batch_size, config.labeled_batch_size)
    else:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, config.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_sampler=batch_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)

    evaldir = os.path.join(datadir, config.eval_subdir)
    evalset = torchvision.datasets.ImageFolder(evaldir,eval_transform)
    eval_loader = torch.utils.data.DataLoader(evalset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=2*config.workers,
                                              pin_memory=True,
                                              drop_last=False)
    return train_loader, eval_loader

def create_loss_fn(config):
    if config.loss == 'soft':
        if torch.__version__ == '0.4.0':
            criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduce=False)
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='none')
    return criterion

def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer

def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps=="":
            return None
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'exp-warmup':
        lr_lambda = exp_warmup(config.rampup_length,
                               config.rampdown_length,
                               config.epochs)
        scheduler = lr_scheduler.LambdaLR(optimizer,
                                          lr_lambda=lr_lambda)
    elif config.lr_scheduler == 'none':
        scheduler = None
    return scheduler

def main(config):
    with SummaryWriter(comment='_{}_{}{}'.format(config.arch,config.dataset,config.model)) as writer:
        print("PyTorch version: {}".format(torch.__version__))
        #writer = None
        if config.dataset == 'cifar10':
            dataset_config = datasets.cifar10()
        elif config.dataset == 'cifar100':
            dataset_config = datasets.cifar100()
        num_classes = dataset_config.pop('num_classes')
        train_loader, eval_loader = create_data_loaders(**dataset_config, config=config)

        dummy_input = (torch.randn(10,3,32,32),)
        net = arch[config.arch](num_classes)
        if writer is not None:
            writer.add_graph(net, dummy_input)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = create_loss_fn(config)
        if config.is_parallel:
            net = torch.nn.DataParallel(net).to(device)
        else:
            device = 'cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu'
            net = net.to(device)
        optimizer = create_optim(net.parameters(), config)
        scheduler = create_lr_scheduler(optimizer, config)

        if config.model == 'ema':
            net2 = arch[config.arch](num_classes)
            net2 = net2.to(device)
            trainer = consTrainer.Trainer(net, net2, optimizer, criterion, device, config, writer)
        elif config.model == 'pi':
            trainer = consTrainer.Trainer(net, net, optimizer, criterion, device, config, writer)
        trainer.loop(config.epochs, train_loader, eval_loader, scheduler=scheduler, print_freq=config.print_freq)
