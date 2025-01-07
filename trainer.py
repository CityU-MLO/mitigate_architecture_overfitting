
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from torch.nn import functional as F
from utils import AverageMeter, DiffAugment


class Trainer(object):

    def __init__(self, cfg, model, optimizer, summary_writer=None, print_freq=1, is_cuda=True, tau=1,
                 max_epoch=10, scheduler=None, attacker=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.iter = 0
        self.print_freq = print_freq
        self.is_cuda = is_cuda
        self.max_epoch = max_epoch
        self.distill_tau = tau
        self.scheduler = scheduler
        self.attacker = attacker

    def train(self, epoch, data_loader):
        self.model.train()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        end = time.time()

        for i, data in enumerate(data_loader):
            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()

            x = DiffAugment(x, aug_mode=self.cfg.aug_mode, schedule=False, epoch=epoch, total_epoch=self.max_epoch)
            self.optimizer.zero_grad()
            loss, pred = self.ce_loss(x, y)
            loss.backward()
            self.optimizer.step()


            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(torch.sum(torch.argmax(pred, dim=-1) == y if y.dim() == 1 else torch.argmax(y, dim=-1)).float() / x.size(0), x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if self.scheduler is not None:
                if self.cfg.strategy is not None:
                    if (epoch+1) <= (self.cfg.max_epoch-self.cfg.base_epoch):
                        step = (epoch % (self.cfg.base_epoch//2))*len(data_loader) + (i+1)
                        if (epoch+1) % (self.cfg.base_epoch//2) == 0:
                            self.scheduler.max_lr *= 0.8
                    else:
                        step = (epoch-(self.cfg.max_epoch-self.cfg.base_epoch))*len(data_loader) + (i+1)
                        if (epoch+1)-(self.cfg.max_epoch-self.cfg.base_epoch) == 0:
                            self.scheduler.max_lr *= 0.1

                    self.scheduler.step(step)
                else:
                    self.scheduler.step()

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('loss_iter', loss_meter.val, self.iter)
                self.summary_writer.add_scalar('acc_iter', acc_meter.val, self.iter)

            if (i + 1) % self.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}] Time:[{:.3f}] " \
                        "Loss:[{:.3f}/{:.3f}]" \
                        "Acc:[{:.3f}/{:.3f}]".format(
                    epoch, i + 1, len(data_loader), batch_time.val,
                    loss_meter.val, loss_meter.avg,
                    acc_meter.val, acc_meter.avg)
                print(p_str)

            self.iter += 1
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('loss_epoch', loss_meter.avg, epoch)
            self.summary_writer.add_scalar('acc_epoch', acc_meter.avg, epoch)

    def kd_train(self, epoch, data_loader, teacher_model, adv=False):
        def fn_kd_loss(pred, teacher_pred):
            kd_loss = F.kl_div(F.log_softmax(pred / self.cfg.kd_temp, dim=-1),
                               F.softmax(teacher_pred / self.cfg.kd_temp, dim=-1),
                               reduction='batchmean')
            return kd_loss

        self.model.train()
        teacher_model.eval()
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        end = time.time()

        for i, data in enumerate(data_loader):
            x, y = data
            if self.is_cuda:
                x = x.cuda()
                y = y.cuda()

            x = DiffAugment(x, aug_mode=self.cfg.aug_mode, schedule=False, epoch=epoch, total_epoch=self.max_epoch)

            ce_loss, pred = self.ce_loss(x, y)

            with torch.no_grad():
                teacher_pred = teacher_model(x)
                teacher_pred = teacher_pred.detach()

            kd_loss = fn_kd_loss(pred, teacher_pred)
            loss = kd_loss * (self.cfg.kd_weight * self.cfg.kd_temp**2) + ce_loss * (1-self.cfg.kd_weight)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(torch.sum(torch.argmax(pred, dim=-1) == y if y.dim() == 1 else torch.argmax(y, dim=-1)).float() / x.size(0), x.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if self.scheduler is not None:
                if self.cfg.strategy is not None:
                    if (epoch + 1) <= (self.cfg.max_epoch - self.cfg.base_epoch):
                        step = (epoch % (self.cfg.base_epoch // 2)) * len(data_loader) + (i + 1)
                        if (epoch + 1) % (self.cfg.base_epoch // 2) == 0:
                            self.scheduler.max_lr *= 0.8
                    else:
                        step = (epoch - (self.cfg.max_epoch - self.cfg.base_epoch)) * len(data_loader) + (i + 1)
                        if (epoch + 1) - (self.cfg.max_epoch - self.cfg.base_epoch) == 0:
                            self.scheduler.max_lr *= 0.1

                    self.scheduler.step(step)
                else:
                    self.scheduler.step()

            if self.summary_writer is not None:
                self.summary_writer.add_scalar('loss_iter', loss_meter.val, self.iter)
                self.summary_writer.add_scalar('acc_iter', acc_meter.val, self.iter)

            if (i + 1) % self.print_freq == 0:
                p_str = "Epoch:[{:>3d}][{:>3d}|{:>3d}] Time:[{:.3f}] " \
                        "Loss:[{:.3f}/{:.3f}]" \
                        "Acc:[{:.3f}/{:.3f}]".format(
                    epoch, i + 1, len(data_loader), batch_time.val,
                    loss_meter.val, loss_meter.avg,
                    acc_meter.val, acc_meter.avg)
                print(p_str)

            self.iter += 1
        if self.summary_writer is not None:
            self.summary_writer.add_scalar('loss_epoch', loss_meter.avg, epoch)
            self.summary_writer.add_scalar('acc_epoch', acc_meter.avg, epoch)


    def ce_loss(self, x, y):
        pred = self.model(x)
        loss = torch.nn.CrossEntropyLoss()(pred, torch.softmax(y.div_(self.distill_tau), dim=-1))
        # loss = torch.nn.CrossEntropyLoss()(pred, y if y.dim() == 1 else torch.softmax(y/self.distill_tau, dim=-1))
        return loss, pred

    @staticmethod
    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        if target.dim() > 1:
            target = torch.argmax(target, dim=-1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def reset(self):
        self.iter = 0

    def close(self):
        self.iter = 0
        if self.summary_writer is not None:
            self.summary_writer.close()


