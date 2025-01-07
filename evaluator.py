from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

import torch
from torch.nn import functional as F
from utils import nfr


class Evaluator(object):
    def __init__(self, model, is_cuda=True, verbose=True):
        super(Evaluator, self).__init__()
        self.model = model
        self.is_cuda = is_cuda
        self.verbose = verbose
        self.ensemble_buffer = None

    def evaluate(self, data_loader, print_freq=1):
        self.model.eval()
        correct, total = 0, 0
        start = time.time()
        for i, data in enumerate(data_loader):
            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()

            pred = self.model(x)
            pred_y = pred.argmax(1)
            correct += pred_y.eq(y).sum().item()
            total += len(y)
            if self.verbose and (i + 1) % print_freq == 0:
                p_str = "[{:3d}|{:3d}] using {:.3f}s ...".format(
                    i + 1, len(data_loader), time.time() - start)
                print(p_str)
        return float(correct)/total


    def ensemble_evaluate(self, data_loader, epoch):
        self.model.eval()
        correct, total = 0, 0
        start = time.time()
        preds = []
        targets = []
        for i, data in enumerate(data_loader):
            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()

            pred = self.model(x)
            preds.append(pred.cpu())
            targets.append(y.cpu())
            total += len(y)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        if (epoch+1) in [500, 1000, 1500, 2000, 2500, 3000]:
            if self.ensemble_buffer is None:
                self.ensemble_buffer = preds
            else:
                self.ensemble_buffer = self.ensemble_buffer + preds
        if self.ensemble_buffer is not None:
            preds += self.ensemble_buffer
        correct = preds.argmax(1).eq(targets).sum().item()

        return float(correct)/total


    def krr_evaluate(self, data_loader, proto_img, proto_y, print_freq=1):
        self.model.eval()
        correct, total = 0, 0
        start = time.time()
        proto_img = proto_img.cuda()
        proto_y = proto_y.cuda()
        proto_feat = self.model(proto_img, output_feat=True)
        for i, data in enumerate(data_loader):
            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()

            feat = self.model(x, output_feat=True)
            pred = nfr(feat, proto_feat, proto_y)
            pred_y = pred.argmax(1)
            correct += pred_y.eq(y).sum().item()
            total += len(y)
            if self.verbose and (i + 1) % print_freq == 0:
                p_str = "[{:3d}|{:3d}] using {:.3f}s ...".format(
                    i + 1, len(data_loader), time.time() - start)
                print(p_str)
        return float(correct) / total
