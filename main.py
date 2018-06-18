import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import get_loader
from solver import Solver
import argparse


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    
    cudnn.benchmark= True
    
    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
#     if not os.path.exists(config.sample_dir):
#         os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    # Data loader    
    train_loader = get_loader(os.path.join(config.image_dir, 'train'), 
                              config.batch_size, 
                              'train', 
                              config.workers, 
                              config.image_size, 
                              config.crop_size)
    print('length of train_loader: {}'.format(len(train_loader)))

    test_loader = get_loader(os.path.join(config.image_dir, 'test'), 
                             config.batch_size, 
                             'test', 
                             config.workers,
                             config.image_size, 
                             config.crop_size)
    print('length of test_loader: {}'.format(len(test_loader)))

                                 
    # Solver for training and testing face-age-classification
    solver = Solver(config, train_loader, test_loader)
    
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.valid()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Model configuration
    parser.add_argument('--crop_size', type=int, default=224, 
                        help='crop size for the CACD2000 dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image resolution')
    parser.add_argument('--mode', default='train', type=str,
                         help='select train/test mode (default: train)')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size')
    parser.add_argument('--workers', default=4, type=int, 
                         help='number of data loading workers (default: 4)')
    parser.add_argument('--pretrain', type=str2bool, default='True',
                         help='pretrain network using imageNet')
#     parser.add_argument('--max_epoch', default=170, type=int,
#                          help='number of total epochs')
    parser.add_argument('--lr', default=0.01, type=float,
                         help='initial learning rate')
    parser.add_argument('--num_iters', type=int, default=375000,
                        help='number of total iterations')
    parser.add_argument('--num_class', type=int, default=6,
                        help='number of class')
    parser.add_argument('--num_iters_decay', type=str, default=10.0,
                        help='number of iter decay')
    
    # testing configurations.
    parser.add_argument('--resume_iters', type=int, default=0,
                        help='number of checkpoint iter')
    # Directories
    parser.add_argument('--log_dir', type=str, default='cacd/logs')
    parser.add_argument('--model_save_dir', type=str, default='cacd/models')
    parser.add_argument('--result_dir', type=str, default='cacd/results')
    parser.add_argument('--image_dir', type=str, default='cacd2000_224')
    parser.add_argument('--age_cls_ckpt', type=str, default='100000.ckpt')
                        
    
    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
#     parser.add_argument('--train_acc_step', type=int, default=2500)
    parser.add_argument('--model_save_step', type=int, default=2000)
    parser.add_argument('--lr_update_step', type=int, default=5000)

    # Miscellaneous
    parser.add_argument('--use_visdom', type=str2bool, default='True')
    parser.add_argument('--model_name', default='resnet50', type=str,
                         help='model name')
    parser.add_argument('--gpu_ids', default='0', type=str,
                         help='id(s) for CUDA_VISIBLE_DEVICES')

    config = parser.parse_args()
    print(config)
    
    # Set visible GPUs 
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids
    
    main(config)
