from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse


def parse_args():
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='Train adversal attack network')
    parser.add_argument('--exp_name',           dest='exp_name', 
                        type=str,               default='debug', 
                        help='exp name used to construct output dir')
    parser.add_argument('--snap_dir',           dest='snap_dir', 
                        type=str,               default='snapshots', 
                        help='directory to save model')
    parser.add_argument('--log_dir',            dest='log_dir', 
                        type=str,               default='logs', 
                        help='directort to save logs')
    parser.add_argument('--no_log',             dest='no_log',
                        action='store_true',
                        help="if record logs (do not log)")
    # dataset and model settings
    parser.add_argument('--data_name',          dest='data_name', 
                        type=str,               default='mnist', 
                        help='used dataset')
    parser.add_argument('--data_dir',           dest='data_dir', 
                        type=str,               default='data/mnist', 
                        help='data directory')
    parser.add_argument('--distill_data_dir',           dest='distill_data_dir',
                        type=str,               default='distill_data/mnist',
                        help='distilled data directory')
    parser.add_argument('--distill_test_data_dir',           dest='distill_test_data_dir',
                        type=str,               default='distill_data/mnist',
                        help='distilled test data directory')
    parser.add_argument('--model_name',         dest='model_name',
                        type=str,               default='mnist',
                        help='network model')
    parser.add_argument('--norm',         dest='norm',
                        type=str,               default='batchnorm',
                        help='normalization layer', choices=['batchnorm', 'instancenorm', 'groupnorm', 'none'])
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--distill_test', action='store_true')
    parser.add_argument('--ckpt_path', type=str)
    # training settings
    parser.add_argument('--max_epoch',          dest='max_epoch',
                        type=int,               default=1000,
                        help='max train steps')
    parser.add_argument('--base_epoch', dest='base_epoch',
                        type=int, default=1000,
                        help='base train steps')
    parser.add_argument('--optim',               dest='optim',
                        type=str,               default='adam',
                        help='optimizer', choices=['adam', 'sgd', 'lion'])
    parser.add_argument('--lr',                 dest='lr', 
                        type=float,             default=5e-4,
                        help='learning rate')
    parser.add_argument('--batch_size',         dest='batch_size', 
                        type=int,               default=128,
                        help='training batch size')
    parser.add_argument('--seed',               dest='seed', 
                        type=int,               default=0, 
                        help='random seed')
    parser.add_argument('--gpu',                dest="gpus", 
                        type=str,               default="0,1", 
                        help="GPU to be used, default is '0,1' ")
    parser.add_argument('--rand',               dest='randomize', 
                        action='store_true', 
                        help='randomize (not use a fixed seed)')
    parser.add_argument('--tau',         dest='tau',
                        type=float,             default=1.,
                        help='temperature for soft label')
    parser.add_argument('--aug_mode',         dest='aug_mode',
                        type=int,             default=2,
                        help='number of sampled augmentation operations')
    parser.add_argument('--scheduler',         dest='scheduler',
                        type=str,             default='improved',
                        help='scheduler', choices=['improved', 'default'])
    parser.add_argument('--strategy',           dest='strategy',
                        type=str,               default=None,
                        help='training strategy', choices=[None, 'more2less'])
    parser.add_argument('--kd', dest='kd', action='store_true', help='knowledge distillation')
    parser.add_argument('--kd_weight', dest='kd_weight', type=float, default=0.5, help='kd weight')
    parser.add_argument('--kd_temp', dest='kd_temp', type=float, default=1.5, help='kd temperature')
    parser.add_argument('--teacher_model_name', dest='teacher_model_name',
                        type=str,               default='convnetfrepo',
                        help='teahcer network model')
    parser.add_argument('--teacher_ckpt_path', type=str)
    parser.add_argument('--kd_beta', dest='kd_beta', type=float, default=0.1, help='kd beta')
    # print and output settings
    parser.add_argument('--print_freq',         dest='print_freq', 
                        type=int,               default=200,
                        help='print freq')
    parser.add_argument('--save_freq',          dest='save_freq', 
                        type=int,               default=50,
                        help='save checkpint freq')
    # evaluation settings
    parser.add_argument('--eval_model',         dest='eval_model', 
                        type=str,               default=None, 
                        help='evaluation checkpint path')
    parser.add_argument('--eval_samples',       dest='eval_samples', 
                        type=int,               default=10000, 
                        help='num of evaluation samples')
    parser.add_argument('--eval_batch_size',    dest='eval_batch_size', 
                        type=int,               default=256,
                        help='evaluation batch size')
    parser.add_argument('--eval_cpu',           dest='eval_cpu',
                        action='store_true',
                        help="if eval on cpu (do not on cpu)")

    parser.add_argument('--fraction', dest='fraction',
                        type=float, default=1.0,
                        help="fraction of training data")

    parser.add_argument('--fraction_seed', dest='fraction_seed',
                        type=int, default=-1,
                        help="if eval on cpu (do not on cpu)")

    args = parser.parse_args()
    return args


cfg = parse_args()
