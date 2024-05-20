# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from backbone.Timm_Wrapper import TimmModel
from backbone.ResNet18_InstanceNorm import resnet18_IN
from PIL import Image
from torchvision.datasets import CIFAR100
import torch.utils.data as dataUtil
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from datasets.seq_cifar100 import MyCIFAR100, TCIFAR100


class ChunkCIFAR100(ContinualDataset):

    NAME = 'temp-chunk-cifar100'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 100
    N_TASKS = -1
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))])

    def __init__(self, args):
        self.N_TASKS = args.Ntasks
        self.pretrain = args.pretrain
        self.instanceNorm = args.instanceNorm
        self.linear_probe = args.linear_probe
        self.warmup = args.warmup
        self.stratified = args.stratified
        self.num_classes = 100
        self.TRANSFORM = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.Resize(size=224, max_size=None, antialias=None),
                                             #transforms.CenterCrop(size=(224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                  (0.2675, 0.2565, 0.2761))])
        #self.N_CLASSES_PER_TASK = 100/self.N_TASKS
        self.task_num = 0
        super().__init__(args=args)
        
        transform = self.TRANSFORM

        test_transform = transforms.Compose([transforms.Resize(size=224, max_size=None, antialias=None),
                                             transforms.ToTensor(), 
                                             self.get_normalization_transform()])
        

        train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
                                   download=True, transform=test_transform)
        
        #self.train_data = train_dataset
        #self.test_data = test_dataset
        if self.stratified:
            self.train_chunks, self.test_chunks = create_stratified_masks(train_dataset, test_dataset, self)
        else:
            self.train_chunks, self.test_chunks = create_tasks(train_dataset, test_dataset, self) 

    def get_examples_number(self):
        train_dataset = MyCIFAR100(base_path() + 'CIFAR10', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):

        if self.stratified:
            transform = self.TRANSFORM

            test_transform = transforms.Compose([transforms.Resize(size=224, max_size=None, antialias=None),
                                                 transforms.ToTensor(),
                                                 self.get_normalization_transform()])
            

            train_dataset = MyCIFAR100(base_path() + 'CIFAR100', train=True,
                                  download=True, transform=transform)
            
            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
            else:
                test_dataset = TCIFAR100(base_path() + 'CIFAR100',train=False,
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
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        self.i += 100/self.N_TASKS
        self.task_num += 1

        return train_loader, test_loader

    #@staticmethod
    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    #@staticmethod
    def get_backbone(self):
        return TimmModel('resnet18', pretrained=False, num_classes=100, linear_probe=self.linear_probe) 
        

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.5071, 0.4867, 0.4408),
                                (0.2675, 0.2565, 0.2761))
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return ChunkCIFAR100.get_batch_size()

    #@staticmethod
    def get_scheduler(self, model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        if self.warmup:
            def warmupStep(epoch):
                if epoch > 3:
                    return 1
                else:
                    return 0.1
            warmup_schedule = torch.optim.lr_scheduler.LambdaLR(model.opt, lr_lambda=warmupStep)  
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_schedule, scheduler])
        return scheduler


def create_tasks(train_dataset, test_dataset, setting):
    # creates the iid chunk tasks
    #train_indices = np.arange(len(train_dataset))
    #train_chunks = [list(a) for a in np.array_split(train_indices, setting.N_TASKS)]
    #test_indices = np.arange(len(test_dataset))
    #test_chunks = [list(a) for a in np.array_split(test_indices, setting.N_TASKS)]
    train_chunks = torch.utils.data.random_split(train_dataset, [len(train_dataset)//setting.N_TASKS]*setting.N_TASKS)
    test_chunks = torch.utils.data.random_split(test_dataset, [len(test_dataset)//setting.N_TASKS]*setting.N_TASKS)
    return train_chunks, test_chunks


def create_stratified_data(dataset, num_tasks, num_classes):
    indicies = [np.array(range(len(dataset.targets)))[np.array(dataset.targets) == i] for i in range(num_classes)]
    for idxs in indicies:
        np.random.shuffle(idxs)
    indicies = np.array(indicies)
    per_class_chunk_size = (len(dataset) // num_classes) // num_tasks
    return [indicies[:, per_class_chunk_size*i:per_class_chunk_size*(i+1)].reshape(-1) for i in range(num_tasks)]   


def create_test_split_masks(dataset, num_tasks):
    indicies = np.array(range(len(dataset)))
    np.random.shuffle(indicies)
    chunk_size = len(dataset) // num_tasks
    return [indicies[chunk_size*i:chunk_size*(i+1)] for i in range(num_tasks)]


def create_stratified_masks(train_dataset, test_dataset, setting):
    train_mask = create_stratified_data(train_dataset, setting.N_TASKS, setting.num_classes)
    test_mask = create_test_split_masks(test_dataset, setting.N_TASKS)
    print(np.array(test_mask).shape)
    return train_mask, test_mask 





