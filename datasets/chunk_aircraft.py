# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import PIL.Image
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import FGVCAircraft
from torch.utils.data import DataLoader, Dataset
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
import numpy as np
from backbone.Timm_Wrapper import TimmModel
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from datasets.chunk_cifar100 import create_tasks  
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]


class TAircraft(FGVCAircraft):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train="trainval", transform=None,
                 target_transform=None, download=True) -> None:
        self.root = root
        super(TAircraft, self).__init__(root=root, split=train, transform=transform, target_transform=target_transform, download=download)
        self.targets = self._labels

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """

        image_file, target = self._image_files[index], self._labels[index]
        img = PIL.Image.open(image_file).convert("RGB")
        img = transforms.Resize(size=(224,224), max_size=None, antialias=None)(img)
        #img, target = self.data[index], self._labels[index]

        # to return a PIL Image
        #img = Image.fromarray(img, mode='RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MyAircraft(FGVCAircraft):
    """
    Overrides the CIFAR100 dataset to change the getitem function.
    """
    def __init__(self, root, train="trainval", transform=None,
                 target_transform=None, download=True) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        print(root)
        super(MyAircraft, self).__init__(root=root, split=train, transform=transform, target_transform=target_transform, download=download)
        self.targets = self._labels

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """

        image_file, target = self._image_files[index], self._labels[index]
        img = PIL.Image.open(image_file).convert("RGB")
        img = transforms.Resize(size=(224,224), max_size=None, antialias=None)(img)
        #img, target = self.data[index], self._labels[index]

        # to return a PIL Image
        #img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class ChunkAircraft(ContinualDataset):

    NAME = 'chunk-aircraft'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 100
    N_TASKS = 10
    TRANSFORM = transforms.Compose(
            [transforms.RandomRotation(degrees=15),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean_nums,
                                  std_nums)])
    
    def __init__(self, args):
        self.N_TASKS = args.Ntasks
        self.pretrain = args.pretrain
        self.instanceNorm = args.instanceNorm
        self.linear_probe = args.linear_probe
        self.warmup = args.warmup
        self.stratified = args.stratified
        self.num_classes = 100
        if self.pretrain:
            self.TRANSFORM = transforms.Compose([transforms.RandomRotation(degrees=15),
                                                 transforms.RandomHorizontalFlip(),
                                                 #transforms.CenterCrop(size=(224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean_nums,
                                                                      std_nums)])
        #self.N_CLASSES_PER_TASK = 100/self.N_TASKS
        self.task_num = 0
        super().__init__(args=args)

        transform = self.TRANSFORM
        
        if self.pretrain:
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 self.get_normalization_transform()])
        else:
            test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyAircraft(base_path() + 'FGVCAircraft', train='trainval',
                                  download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = TAircraft(base_path() + 'FGVCAircraft', train="test",
                                   download=True, transform=test_transform)

        #self.train_data = train_dataset
        #self.test_data = test_dataset
        if self.stratified:
            self.train_chunks, self.test_chunks = create_stratified_masks(train_dataset, test_dataset, self)
        else:
            self.train_chunks, self.test_chunks = create_tasks(train_dataset, test_dataset, self)
        


    def get_examples_number(self):
        train_dataset = MyAircraft(base_path() + 'FGVCAircraft', train=True,
                                  download=True)
        return len(train_dataset.data)

    def get_data_loaders(self):
        if self.stratified:
            transform = self.TRANSFORM

            if self.pretrain:
                test_transform = transforms.Compose([transforms.ToTensor(),
                                                     self.get_normalization_transform()])
            else:
                test_transform = transforms.Compose([transforms.ToTensor(), self.get_normalization_transform()])

            train_dataset = MyAircraft(base_path() + 'FGVCAircraft', train="trainval",
                                  download=True, transform=transform)

            if self.args.validation:
                train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
            else:
                test_dataset = TAircraft(base_path() + 'FGVCAircraft', train="test",
                                       download=True, transform=test_transform)

            train_mask, test_mask = self.train_chunks[self.task_num], self.test_chunks[self.task_num]

            train_dataset._image_files = np.array(train_dataset._image_files)[train_mask]
            test_dataset._image_files = np.array(test_dataset._image_files)[test_mask]

            train_dataset._labels = np.array(train_dataset._labels)[train_mask]
            test_dataset._labels = np.array(test_dataset._labels)[test_mask]

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

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), ChunkAircraft.TRANSFORM])
        return transform

    def get_backbone(self):
        if self.pretrain:
            return TimmModel('resnet18', pretrained=self.pretrain, num_classes=100, linear_probe=self.linear_probe)
        return resnet18(100)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(mean_nums,
                                         std_nums) 
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(mean_nums,
                                std_nums)
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return ChunkAircraft.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        return scheduler


def create_stratified_data(dataset, num_tasks, num_classes):
    indicies = [np.array(range(len(dataset._labels)))[np.array(dataset._labels) == i] for i in range(num_classes)]
    min_len = len(dataset)
    indicies = [idxs[:60] for idxs in indicies]
    for idxs in indicies:
        if len(idxs) < min_len:
            min_len = len(idxs)
        np.random.shuffle(idxs)
    indicies = [idxs[:min_len] for idxs in indicies]
    per_class_chunk_size = (len(indicies)*min_len // num_classes) // num_tasks
    indicies = np.array(indicies)
    return [indicies[:, per_class_chunk_size*i:per_class_chunk_size*(i+1)].reshape(-1) for i in range(num_tasks)]


def create_test_split_masks(dataset, num_tasks):
    indicies = np.array(range(len(dataset)))
    np.random.shuffle(indicies)
    chunk_size = len(dataset) // num_tasks
    masks = []
    for i in range(num_tasks-1):
        masks.append(indicies[chunk_size*i:chunk_size*(i+1)])
    masks.append(indicies[chunk_size*(num_tasks-1):])
    return masks


def create_stratified_masks(train_dataset, test_dataset, setting):
    train_mask = create_stratified_data(train_dataset, setting.N_TASKS, setting.num_classes)
    test_mask = create_test_split_masks(test_dataset, setting.N_TASKS)
    print("trainShape:"+str(np.array(train_mask).shape))
    print("testShape:"+str(len(test_mask))+", "+str(len(test_mask[0]))+", "+str(len(test_mask[-1])))
    return train_mask, test_mask


