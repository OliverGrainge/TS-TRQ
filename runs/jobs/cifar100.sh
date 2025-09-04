#!/bin/bash


python train.py runs/configs/resnet/cifar100/resnet18_untrained_cifar100_linear.yaml
python train.py runs/configs/resnet/cifar100/resnet18_cifar100_tlinear.yaml
python train.py runs/configs/resnet/cifar100/resnet18_cifar100_tsvdlinear.yaml
python train.py runs/configs/resnet/cifar100/resnet18_cifar100_linear.yaml