# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from datasets.seq_tinyimagenet import TinyImagenet, MyTinyImagenet  
from datasets.chunk_cifar100 import create_tasks, create_stratified_masks

class ChunkTinyImagenet(ContinualDataset):

    NAME = 'chunk-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 200
    N_TASKS = -1
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4802, 0.4480, 0.3975),
                              (0.2770, 0.2691, 0.2821))])

    def __init__(self, args):
        self.N_TASKS = args.Ntasks
        self.pretrain = args.pretrain
        self.instanceNorm = args.instanceNorm
        self.linear_probe = args.linear_probe
        self.warmup = args.warmup
        self.task_num = 0
        self.stratified = args.stratified
        self.num_classes = 200
        self.full_eval = args.full_eval
        self.single_chunk_eval = args.single_chunk_eval
        super().__init__(args=args)

        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                                       train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                                        train=False, download=True, transform=test_transform)
        
        if self.full_eval:
            test_loader = DataLoader(test_dataset,
                                     batch_size=ChunkTinyImagenet.get_batch_size(), shuffle=False, num_workers=4)
            self.test_loaders.append(test_loader)
        
        if self.stratified:
            self.train_chunks, self.test_chunks = create_stratified_masks(train_dataset, test_dataset, self)
        else:
            self.train_chunks, self.test_chunks = create_tasks(train_dataset, test_dataset, self)

        if self.single_chunk_eval is not None:
            if self.stratified:
                train_dataset = TinyImagenet(base_path() + 'TINYIMG', train=True,
                                          download=True, transform=test_transform)
                train_mask = self.train_chunks[self.single_chunk_eval]

                train_dataset.data = train_dataset.data[train_mask]
                train_dataset.targets = np.array(train_dataset.targets)[train_mask]

            else:
                train_dataset = self.train_chunks[self.single_chunk_eval]

            train_loader = DataLoader(train_dataset,
                                      batch_size=ChunkTinyImagenet.get_batch_size(), shuffle=False, num_workers=4)
            self.test_loaders.append(train_loader)

    def get_data_loaders(self):
        if self.stratified:
            transform = self.TRANSFORM

            test_transform = transforms.Compose(
                [transforms.ToTensor(), self.get_normalization_transform()])

            train_dataset = MyTinyImagenet(base_path() + 'TINYIMG',
                                           train=True, download=True, transform=transform)
            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset,
                                                            test_transform, self.NAME)
            else:
                test_dataset = TinyImagenet(base_path() + 'TINYIMG',
                                            train=False, download=True, transform=test_transform)
            
            train_mask, test_mask = self.train_chunks[self.task_num], self.test_chunks[self.task_num]

            train_dataset.data = train_dataset.data[train_mask]
            test_dataset.data = test_dataset.data[test_mask]

            train_dataset.targets = np.array(train_dataset.targets)[train_mask]
            test_dataset.targets = np.array(test_dataset.targets)[test_mask]

        else:
            train_dataset = self.train_chunks[self.task_num]
            test_dataset = self.test_chunks[self.task_num]

        train_loader = DataLoader(train_dataset,
                              batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset,
                             batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        if (not self.full_eval) and (self.single_chunk_eval is None):
            self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        self.i += 200/self.N_TASKS
        self.task_num += 1

        return train_loader, test_loader

    @staticmethod
    def get_backbone():
        return resnet18(200)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4802, 0.4480, 0.3975),
                                (0.2770, 0.2691, 0.2821))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return ChunkTinyImagenet.get_batch_size()
