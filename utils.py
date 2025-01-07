from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import kornia as K
from tqdm import tqdm

import math

import torch
from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
from torchvision import datasets as D
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class TensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()
        self.transform = transform

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_data_loader(cfg, data_name='mnist', data_dir='data/mnist', batch_size=128, fraction=1., seed=-1,
                    test_batch_size=200, num_workers=0, is_infinite=False):
    if data_name == 'mnist':
        transform_train = T.Compose([T.ToTensor()])
        transform_test = T.Compose([T.ToTensor()])
        train_set = D.MNIST(root=data_dir, train=True, download=True,
                            transform=transform_train)
        test_set = D.MNIST(root=data_dir, train=False, download=True,
                           transform=transform_test)
        img_size, num_class = 28, 10
    elif data_name == 'cifar10':

        if cfg.zca:
            transform_train = T.Compose([T.ToTensor()])
            transform_test = T.Compose([T.ToTensor()])
        else:
            transform_train = T.Compose([T.ToTensor(),
                                         T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                         ])
            transform_test = T.Compose([T.ToTensor(),
                                        T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                                        ])

        train_set = D.CIFAR10(root=data_dir, train=True, download=True,
                              transform=transform_train)
        test_set = D.CIFAR10(root=data_dir, train=False, download=True,
                             transform=transform_test)
        img_size, num_class = 32, 10
    elif data_name == 'cifar100':

        if cfg.zca:
            transform_train = T.Compose([T.ToTensor()])
            transform_test = T.Compose([T.ToTensor()])
        else:
            transform_train = T.Compose([T.ToTensor(),
                                         T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                         ])
            transform_test = T.Compose([T.ToTensor(),
                                        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                        ])

        train_set = D.CIFAR100(root=data_dir, train=True, download=True,
                              transform=transform_train)
        test_set = D.CIFAR100(root=data_dir, train=False, download=True,
                             transform=transform_test)
        img_size, num_class = 32, 100
    elif data_name == 'stl10':
        transform_train = T.Compose([T.RandomCrop(96, padding=4),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(), ])
        transform_test = T.Compose([T.ToTensor()])
        train_set = D.STL10(root=data_dir, split='train', download=True,
                            transform=transform_train)
        test_set = D.STL10(root=data_dir, split='test', download=True,
                           transform=transform_test)
        img_size, num_class = 96, 10
    elif data_name == 'imagenet-sub':
        # if cfg.zca:
        #     transform_train = T.Compose([T.ToTensor()])
        #     transform_test = T.Compose([T.ToTensor()])
        # else:[0.4759, 0.4481, 0.3926], [0.2763, 0.2687, 0.2813]
        transform_train = T.Compose([T.ToTensor(),
                                         T.Normalize(mean=[0.4759, 0.4481, 0.3926], std=[0.2763, 0.2687, 0.2813])
                                         ])
        transform_test = T.Compose([T.ToTensor(),
                                        T.Normalize(mean=[0.4759, 0.4481, 0.3926], std=[0.2763, 0.2687, 0.2813])
                                        ])

        train_set = D.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        test_set = D.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        img_size, num_class = 64, 200
    else:
        raise ValueError('invalid dataset, current only support {}'.format(
            "mnist, cifar10, stl10, imagenet-sub"))

    if cfg.zca:
        images = []
        labels = []
        print("Train ZCA")
        zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
        if data_name != 'imagenet-sub':
            for i in tqdm(range(len(train_set))):
                im, lab = train_set[i]
                images.append(im)
                labels.append(lab)
            images = torch.stack(images, dim=0).cuda()
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")
            zca.fit(images)
            # save zca matrix
            # torch.save({'mean': zca.mean_vector, 'transform': zca.transform_matrix, 'transform_inv': zca.transform_inv},
            #            os.path.join(data_dir, 'zca.pt'))
        else:
            # load zca matrix
            zca_dict = torch.load(os.path.join(data_dir, 'zca.pt'))
            zca.mean_vector = zca_dict['mean']
            zca.transform_matrix = zca_dict['transform']
            zca.transform_inv = zca_dict['transform_inv']
            zca.fitted = True

        # zca_images = zca(images).to("cpu")
        # if data_name == 'mnist':
        #     transform_train = None
        # elif data_name == 'cifar10':
        #     transform_train = None
        # elif data_name == 'stl10':
        #     transform_train = None
        # elif data_name == 'imagenet-sub':
        #     transform_train = None
        # train_set = TensorDataset(zca_images, labels, transform=transform_train)

        test_images = []
        test_labels = []
        print("Test ZCA")
        for i in tqdm(range(len(test_set))):
            im, lab = test_set[i]
            test_images.append(im)
            test_labels.append(lab)

        test_images = torch.stack(test_images, dim=0).cuda()
        test_labels = torch.tensor(test_labels, dtype=torch.long, device="cpu")
        if data_name == 'imagenet-sub':
            zca_test_images = zca(test_images.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous().to("cpu")
        else:
            zca_test_images = zca(test_images).to("cpu")
        test_set = TensorDataset(zca_test_images, test_labels)
        cfg.zca_trans = zca

    if fraction < 1:
        images = []
        labels = []
        indices_class = [[] for c in range(num_class)]
        for i in tqdm(range(len(train_set))):
            im, lab = train_set[i]
            images.append(im)
            labels.append(lab)
        for i, lab in tqdm(enumerate(labels)):
            indices_class[lab].append(i)
        images = torch.stack(images, dim=0).cpu()
        labels = torch.tensor(labels, dtype=torch.long, device="cpu")

        def get_images(c, n, seed=-1):  # get random n images from class c
            if seed != -1:
                np.random.seed(seed)
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images[idx_shuffle]

        ipc = int(fraction * len(labels) / num_class)
        label_down = torch.tensor(np.array([np.ones(ipc, dtype=np.int_) * i for i in range(num_class)]),
                                  dtype=torch.long).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        image_down = torch.randn(size=(num_class * ipc, 3, img_size, img_size), dtype=torch.float)

        for c in range(num_class):
            image_down[c * ipc:(c + 1) * ipc] = get_images(c, ipc, seed=seed).clone()
        train_set = TensorDataset(image_down, label_down)

    if is_infinite:
        train_loader = InfiniteDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                          pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, num_class, img_size, train_set, test_set


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(model, path):
    if not os.path.isfile(path):
        raise IOError('model: {} is non-exists'.format(path))
    if hasattr(model, 'module'):
        module = model.module
    else:
        module = model
    checkpoint = torch.load(path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    module.load_state_dict(state_dict, strict=False)
    print('Params Loaded from: {}'.format(path))


class DistilledData(Dataset):
    def __init__(self, data_dir, transform=None, original_data=None):
        super(DistilledData, self).__init__()
        self.x = torch.load(os.path.join(data_dir, 'images_best.pt'))
        self.y = torch.load(os.path.join(data_dir, 'labels_best.pt'))
        if original_data is not None:
            self.x = torch.cat((self.x, original_data[0].permute(0, 3, 1, 2)), dim=0)
            self.y = torch.cat((self.y, original_data[1]), dim=0)
        assert len(self.x) == len(self.y), 'The length of images is not the same as that of labels'
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img, label = self.x[idx], self.y[idx]
        img = img
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_distilled_data_loader(data_dir='distilled_data/mnist', batch_size=128, num_workers=4, mode='train',
                              original_data=None, is_infinite=False):
    dataset = DistilledData(data_dir, original_data=original_data)

    if is_infinite:
        data_loader = InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True if mode == 'train' else False,
                                         num_workers=num_workers, pin_memory=True)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True if mode == 'train' else False,
                                 num_workers=num_workers, pin_memory=True)

    return data_loader, dataset, torch.std(dataset.x).item()


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class ParamDiffAug():
    def __init__(self, strength=1):
        self.aug_mode = 'S'  #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1 + 0.2 * strength
        self.ratio_rotate = 15.0 * strength
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 * strength  # the size would be 0.5x0.5
        self.ratio_noise = 0.05 * strength
        self.brightness = 1.0 * strength
        self.saturation = 2.0 * strength
        self.contrast = 0.5 * strength


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, aug_mode=2, schedule=False, epoch=-1, total_epoch=1000):
    if schedule:
        strength = 1 - np.floor(epoch/50) * 50 / total_epoch
    else:
        strength = 1
    param = ParamDiffAug(strength)
    param.aug_mode = aug_mode
    strategy = 'color_crop_cutout_flip_scale_rotate'
    seed = -1
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        elif isinstance(param.aug_mode, int):
            pbties = np.array(strategy.split('_'))
            set_seed_DiffAug(param)
            indices = torch.randperm(len(pbties))[:param.aug_mode].numpy()
            ps = pbties[indices]
            for p in ps:
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

def random_sample(dataset, num_classes=10, ipc=10):
    """Randomly sample a subset of the dataset."""
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    im_size = dataset[0][0].shape[-2:]
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(sample[1].item())

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")

    # create validation set
    val_label = torch.tensor(np.array([np.ones(ipc, dtype=np.int_) * i for i in range(num_classes)]),
                             dtype=torch.long, requires_grad=False).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
    val_image = torch.randn(size=(num_classes * ipc, 3, im_size[0], im_size[1]), dtype=torch.float)

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    for c in range(num_classes):
        val_image.data[c * ipc:(c + 1) * ipc] = get_images(c, ipc).detach().data

    return val_image, val_label


def nfr(x_target, x_proto, y_proto, reg=1e-6, return_kernel=False):
    k_pp = x_proto @ x_proto.T
    k_tp = x_target @ x_proto.T
    regularization = abs(reg) * torch.trace(k_pp) * torch.eye(k_pp.shape[0], device=x_proto.device) / k_pp.shape[0]
    k_pp_reg = (k_pp + regularization)
    pred = k_tp @ torch.linalg.solve(k_pp_reg, y_proto)
    if return_kernel:
        return pred, (k_pp, k_tp)
    return pred


class CosineAnnealingWarmupRestarts(LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            if epoch == 1999:
                self.cycle_mult = 2
                self.warmup_steps *= 2
                print('Current cycle steps are {}'.format(int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps))
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr