#!/bin/bash

python train.py configs/vit_cifar100_l1_tsvd/reg1.yaml
python train.py configs/vit_cifar100_l1_tsvd/reg2.yaml
python train.py configs/vit_cifar100_l1_tsvd/reg3.yaml
python train.py configs/vit_cifar100_l1_tsvd/reg4.yaml
python train.py configs/vit_cifar100_l1_tsvd/reg5.yaml

python train.py configs/vit_cifar100_l2_tsvd/reg1.yaml
python train.py configs/vit_cifar100_l2_tsvd/reg2.yaml
python train.py configs/vit_cifar100_l2_tsvd/reg3.yaml
python train.py configs/vit_cifar100_l2_tsvd/reg4.yaml
python train.py configs/vit_cifar100_l2_tsvd/reg5.yaml

python train.py configs/vit_cifar100_l1_tsvdt/reg1.yaml
python train.py configs/vit_cifar100_l1_tsvdt/reg2.yaml
python train.py configs/vit_cifar100_l1_tsvdt/reg3.yaml
python train.py configs/vit_cifar100_l1_tsvdt/reg4.yaml
python train.py configs/vit_cifar100_l1_tsvdt/reg5.yaml

python train.py configs/vit_cifar100_l2_tsvdt/reg1.yaml
python train.py configs/vit_cifar100_l2_tsvdt/reg2.yaml
python train.py configs/vit_cifar100_l2_tsvdt/reg3.yaml
python train.py configs/vit_cifar100_l2_tsvdt/reg4.yaml
python train.py configs/vit_cifar100_l2_tsvdt/reg5.yaml
