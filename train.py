from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import pprint

from torch.backends import cudnn
from config import cfg
from model import create_model, set_drop_path
from trainer import Trainer
from evaluator import Evaluator
from utils import *
from optim import SAM, Lion


if __name__ == '__main__':
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    if not cfg.randomize:
        # set fixed seed
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
    # log_path = os.path.join(cfg.log_dir, cfg.exp_name)
    # mkdir_if_missing(log_path)
    snap_path = os.path.join(cfg.snap_dir, cfg.exp_name)
    mkdir_if_missing(snap_path)

    summary_writer = None
    # if not cfg.no_log:
    #     log_name = cfg.exp_name + "_log_" + \
    #                strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '.txt'
    #     sys.stdout = Logger(os.path.join(log_path, log_name))
    #     summary_writer = SummaryWriter(log_dir=log_path)

    print("Input Args: ")
    pprint.pprint(cfg)
    train_loader, test_loader, num_classes, img_size, train_set, test_set = get_data_loader(
        cfg, data_name=cfg.data_name, data_dir=cfg.data_dir, batch_size=cfg.batch_size,
        test_batch_size=cfg.eval_batch_size, num_workers=4)

    if cfg.zca:
        zca_trans = cfg.zca_trans
    else:
        zca_trans = None

    distill_loader, distill_dataset, distill_std = get_distilled_data_loader(data_dir=cfg.distill_data_dir,
                                                                             batch_size=cfg.batch_size, num_workers=4)

    scheduler = None
    model = create_model(name=cfg.model_name, num_classes=num_classes, channel=3, norm=cfg.norm,
                         im_size=(64, 64) if cfg.data_name == 'imagenet-sub' else (32, 32),
                         # net_depth=4 if cfg.data_name == 'imagenet-sub' else 3
                         )

    if cfg.kd:
        teacher_model = create_model(name=cfg.teacher_model_name, num_classes=num_classes, channel=3, norm=cfg.norm,
                                     im_size=(64, 64) if cfg.data_name == 'imagenet-sub' else (32, 32),
                                     net_depth=4 if cfg.data_name == 'imagenet-sub' else 3
                                     )
        teacher_model.load_state_dict(torch.load(cfg.teacher_ckpt_path)['state_dict'])


    if cfg.optim == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0005)
    elif cfg.optim == 'lion':
        optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=0.005)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=0.0005, momentum=0.9)


    if cfg.strategy is not None:
        if cfg.scheduler == 'default':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=0.01 * cfg.lr,
                T_max=1000 * len(distill_loader))
        else:
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=cfg.base_epoch * len(distill_loader),
                                                      cycle_mult=1.0, max_lr=cfg.lr, min_lr=0,
                                                      warmup_steps=cfg.base_epoch // 10 * len(distill_loader),
                                                      gamma=1.0)
    else:
        if cfg.scheduler == 'default':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[
                    500 * len(distill_loader),
                ], gamma=0.1)
        else:
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=cfg.max_epoch * len(distill_loader),
                                                      cycle_mult=1.0, max_lr=cfg.lr, min_lr=0.,
                                                      warmup_steps=cfg.max_epoch // 10 * len(distill_loader),
                                                      gamma=1.0)

    is_cuda = False
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if cfg.kd:
            teacher_model = nn.DataParallel(teacher_model)
    if torch.cuda.is_available():
        model = model.cuda()
        if cfg.kd:
            teacher_model = teacher_model.cuda()
        is_cuda = True

    trainer = Trainer(cfg=cfg, model=model, optimizer=optimizer, summary_writer=summary_writer, is_cuda=True,
                      print_freq=cfg.print_freq, max_epoch=cfg.max_epoch, tau=cfg.tau, scheduler=scheduler)
    evaluator = Evaluator(model=model, is_cuda=is_cuda, verbose=False)

    best_accuracy = 0
    trainer.reset()
    for epoch in range(cfg.max_epoch):
        if cfg.strategy == 'more2less':
            if epoch in range(0, cfg.max_epoch - cfg.base_epoch, cfg.base_epoch // 2):
                set_drop_path(model, 1 - 0.13 * np.floor(
                    epoch / (cfg.base_epoch // 2)) if 'resnet50' in cfg.model_name else 1 - 0.1 * np.floor(
                    epoch / (cfg.base_epoch // 2)))
            elif epoch == cfg.max_epoch - cfg.base_epoch:
                set_drop_path(model, 0.8)

        if cfg.kd:
            trainer.kd_train(epoch, distill_loader, teacher_model=teacher_model)
            # trainer.kd_train(epoch, train_loader, teacher_model=teacher_model)
        else:
            trainer.train(epoch, distill_loader)
            # trainer.train(epoch, train_loader)

        if cfg.save_freq < 1:
            save_current = False
        else:
            save_current = (epoch + 1) % cfg.save_freq == 0 \
                           or epoch == 0 or epoch == cfg.max_epoch - 1
        if save_current:
            with torch.no_grad():
                test_acc = evaluator.evaluate(test_loader)

            if summary_writer is not None:
                summary_writer.add_scalar('test_acc', test_acc, epoch)
            print("epoch {:3d} evaluated".format(epoch))
            print("test accuracy: {:.4f}".format(test_acc))
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                if hasattr(model, 'module'):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                dict_to_save = {'state_dict': state_dict,
                                'epoch': epoch + 1}
                fpath = os.path.join(snap_path, 'checkpoint_best.pth')
                torch.save(dict_to_save, fpath)

            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            dict_to_save = {'state_dict': state_dict,
                            'epoch': epoch + 1}
            fpath = os.path.join(snap_path, 'checkpoint_last.pth')
            torch.save(dict_to_save, fpath)
    print('best accuracy: {:.4f}'.format(best_accuracy))
    trainer.close()
