# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR10
import torch.utils.data as dataUtil
from torch.utils.data import DataLoader, Dataset
import numpy as np

from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from datasets.seq_cifar10 import MyCIFAR10, TCIFAR10
from datasets.chunk_cifar100 import create_stratified_masks


class ChunkCIFAR10(ContinualDataset):

    NAME = 'chunk-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = None
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))])
    
    def __init__(self, args):
        self.N_TASKS = args.Ntasks
        #self.N_CLASSES_PER_TASK = 100/self.N_TASKS
        self.task_num = 0
        self.stratified = args.stratified
        self.num_classes = 10
        self.warmup = args.warmup
        self.full_eval = args.full_eval
        self.single_chunk_eval = args.single_chunk_eval
            
        super().__init__(args=args)
        
        transform = self.TRANSFORM

        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=test_transform)
        
        if self.full_eval:
            test_loader = DataLoader(test_dataset,
                                     batch_size=ChunkCIFAR10.get_batch_size(), shuffle=False, num_workers=4)
            self.test_loaders.append(test_loader)
    
        if self.stratified:
            self.train_chunks, self.test_chunks = create_stratified_masks(train_dataset, test_dataset, self)
        else:
            self.train_chunks, self.test_chunks = create_tasks(train_dataset, test_dataset, self)
        
        if self.single_chunk_eval is not None:
            if self.stratified:
                train_dataset = TCIFAR10(base_path() + 'CIFAR10',train=True,
                                          download=True, transform=test_transform)
                train_mask = self.train_chunks[self.single_chunk_eval]

                train_dataset.data = train_dataset.data[train_mask]
                train_dataset.targets = np.array(train_dataset.targets)[train_mask]

            else:
                train_dataset = self.train_chunks[self.single_chunk_eval]

            train_loader = DataLoader(train_dataset,
                                      batch_size=ChunkCIFAR10.get_batch_size(), shuffle=False, num_workers=4)
            self.test_loaders.append(train_loader) 

    def get_data_loaders(self):

        if self.stratified:
            transform = self.TRANSFORM

            test_transform = transforms.Compose(
                [transforms.ToTensor(), self.get_normalization_transform()])

            train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                      download=True, transform=transform)
            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
            else:
                test_dataset = TCIFAR10(base_path() + 'CIFAR10',train=False,
                                       download=True, transform=test_transform)
            
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
        self.i += 10/self.N_TASKS
        self.task_num += 1

        return train_loader, test_loader

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), ChunkCIFAR10.TRANSFORM])
        return transform

    def get_backbone(self):
        return resnet18(10)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    def get_scheduler(self, model, args):
        if self.warmup:
            model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=1.0, verbose=False)
            def warmupStep(epoch):
                if epoch > 3:
                    return 1
                else:
                    return 0.01
            warmup_schedule = torch.optim.lr_scheduler.LambdaLR(model.opt, lr_lambda=warmupStep)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_schedule, scheduler])
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return ChunkCIFAR10.get_batch_size()

def create_tasks(train_dataset, test_dataset, setting):
    # creates the iid chunk tasks
    #train_indices = np.arange(len(train_dataset))
    #train_chunks = [list(a) for a in np.array_split(train_indices, setting.N_TASKS)]
    #test_indices = np.arange(len(test_dataset))
    #test_chunks = [list(a) for a in np.array_split(test_indices, setting.N_TASKS)]
    train_chunks = dataUtil.random_split(train_dataset, [len(train_dataset)//setting.N_TASKS]*setting.N_TASKS)
    test_chunks = dataUtil.random_split(test_dataset, [len(test_dataset)//setting.N_TASKS]*setting.N_TASKS)
    return train_chunks, test_chunks
