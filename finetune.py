import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time


class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        self.model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            # if args.use_cuda:
            #    self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)
        self.infer_data_loader = dataset.infer_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def infer(self):
        self.model.eval()

        print("Calculating avg. inference latency...")
        infer_list = []
        for i, (batch, label) in enumerate(self.infer_data_loader):
            # if args.use_cuda:
            #    batch = batch.cuda()
            pred_start = time.time()
            self.model(Variable(batch))
            pred_end = time.time()
            infer_list.append((pred_end - pred_start) * 1000)
            # print("Inference time: " + str((pred_end - pred_start) * 1000) + "ms")

        infer_avg = sum(infer_list) / len(infer_list)
        print("Avg. inference time after " + str(len(infer_list)) + " iteration(s): " + str(infer_avg) + "ms")

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            # if args.use_cuda:
            #    batch = batch.cuda()
            output = self.model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(self.model.classifier.parameters(), lr=0.0001, momentum=0.9)

        print(id(optimizer))

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    def train_batch(self, optimizer, batch, label, rank_filters):

        # if args.use_cuda:
        #    batch = batch.cuda()
        #    label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters=True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self, prune_degree):
        # Get the accuracy before prunning
        # self.test()
        # self.model.train()

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        prune_degree_perc = (prune_degree / 100)
        num_filters_to_prune_per_iteration = int(number_of_filters * prune_degree_perc)
        # iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = 1

        print("Reduce " + str(prune_degree) + "% of conv filters: " + str(num_filters_to_prune_per_iteration))

        for _ in range(iterations):
            print("Ranking filters.. ")
            rank_start = time.time()
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1
            rank_end = time.time()
            print("Rank time: " + str((rank_end - rank_start)) + "s")
            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            prune_start = time.time()
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=False)

            self.model = model
            # if args.use_cuda:
            #    self.model = self.model.cuda()

            message = str(100 * float(self.total_num_filters()) / number_of_filters) + "%"
            prune_end = time.time()
            print("Prune time: " + str((prune_end - prune_start)) + "s")
            print("Filters prunned", str(message))
            # self.test()
            # print("Fine tuning to recover from prunning iteration.")
            # optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            # self.train(optimizer, epoches=10)

        # print("Finished. Going to fine tune the model a bit more")
        # self.train(optimizer, epoches=15)
        torch.save(model, str("model_pruned_" + str(prune_degree)))
