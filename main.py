import torch
from torch.autograd import Variable
from torchvision import models
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
import finetune


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--infer", dest="infer", action="store_true")
    parser.add_argument("--profile", dest="profile", action="store_true")
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.add_argument("--model", type=str, default="model")
    parser.add_argument("--prune_degree", type=int, default=0)
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    parser.set_defaults(test=False)
    parser.set_defaults(infer=False)
    parser.set_defaults(profile=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args


if __name__ == '__main__':
    args = get_args()

    print(args.prune_degree)
    print(args.model)

    if args.train:
        model = finetune.ModifiedVGG16Model()
    elif args.prune or args.test:
        model = torch.load(str(args.model), map_location=lambda storage, loc: storage)

    if args.use_cuda:
        model = model.cuda()

    print(model)
    print(sys.version)

    fine_tuner = finetune.PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epoches=10)
        torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune(args.prune_degree)

    elif args.test:
        fine_tuner.test()
        fine_tuner.infer()
