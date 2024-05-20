# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import get_dataset

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from copy import deepcopy
from torch.optim import SGD
from models.lwf import modified_kl_div, smooth


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' online EWC.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--e_lambda', type=float, required=True,
                        help='lambda weight for IMM')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, default=2,
                        help='Temperature of the softmax function.')
    return parser


class IMM(ContinualModel):
    NAME = 'imm'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(IMM, self).__init__(backbone, loss, args, transform)
        self.task_num = 0
        self.soft = nn.Softmax(dim=1)
        self.checkpoint = None
        self.fish = None
        self.avg_model = deepcopy(self.net).to(self.device)
        self.eval_model = deepcopy(self.net).to(self.device)
        self.dataset = get_dataset(args)
        # Instantiate buffers
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.class_means = None
        self.old_net = None
        self.current_task = 0

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = ((self.net.get_params() - self.checkpoint) ** 2).sum()
            return penalty

    def end_task(self, dataset):
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        self.current_task += 1

        self.task_num += 1
        fish = 1e-8*torch.ones_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            inputs = data[0]
            labels = data[1]
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                prob = self.soft(output)
                log_prob = torch.log(prob)
                prob_dist = torch.distributions.Categorical(prob)
                y = prob_dist.sample()
                #loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                #                    reduction='none')
                for i in range(log_prob.shape[0]):
                    avg_loss = log_prob[i][y]
                    self.opt.zero_grad()
                    avg_loss.backward()
                    fish += self.net.get_grads() ** 2

        self.opt.zero_grad()
        fish /= (len(dataset.train_loader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish += fish

        with torch.no_grad():
            next = 0
            for avg_param, param in zip(self.avg_model.parameters(), self.net.parameters()):
                avg_update = (fish[next:next+param.numel()] * param.view(-1)).view(param.shape)
                avg_param.data.mul_(self.task_num - 1).add_(avg_update).mul_(1 / self.task_num)
                next += param.numel()

            next = 0
            self.eval_model = deepcopy(self.net).to(self.device)
            for avg_param, eval_param in zip(self.avg_model.parameters(), self.eval_model.parameters()):
                eval_param.copy_((self.task_num/self.fish[next:next+eval_param.numel()]*avg_param.data.view(-1)).view(eval_param.shape))
                next += eval_param.numel()

        self.checkpoint = self.net.get_params().data.clone()

    # only used for eval, such that we test using EMA model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            status = self.eval_model.training
            self.eval_model.eval()
            # with torch.no_grad():
            out = self.eval_model(x)
            self.eval_model.train(status)
            return out

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.
        :param inputs: the images to be fed to the network
        :param labels: the ground-truth labels
        :param task_idx: the task index
        :return: the differentiable loss value
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs = self.net(inputs)[:, :ac]
        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :ac]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, pc:ac]
            comb_targets = torch.cat((logits[:, :pc], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        return loss

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):
        # if self.task_num > 0:
        #    for module in self.net.children():
        #        if isinstance(module, nn.BatchNorm2d):
        #            module.eval()
        if self.current_task > 0:
            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.current_task, logits)
        penalty = self.penalty()
        loss += self.args.e_lambda * penalty
        loss.backward()

        self.opt.step()

        return loss.item()


def get_state(net):
    params = []
    for pp in list(net.parameters()):
        params.append(pp.view(-1))
    return torch.cat(params)
