from __future__ import print_function

# Standard lib imports
import os
import time
import argparse
import os.path as osp
from urllib.parse import urlparse

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose, ToTensor, Normalize, Scale

# file imports
from models import resnet
from utils import AverageMeter
from utils.losses import MultiLLFunction
from utils.dataloader import ImageFilelist

# Dataloading settings
parser = argparse.ArgumentParser(
    description='CASENet training routine')

parser.add_argument('--data-root', type=str, default='/home/ebotero/SSD3/contours/SBD',
                    help='root path')
parser.add_argument('--data-txt', type=str, default='train_contours.txt',
                    help='path to train txt file containing image and ground truth paths')
parser.add_argument('--save-folder', type=str, default="weights/")
parser.add_argument('--snapshot', type=str, default='weights/casenet_weights.pth',
                    help='path to weight snapshot file')
parser.add_argument('--dataset', type=str, default='SBD',
                    help='dataset to train')

# Training settings
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--backup-iters', type=int, default=10000,
                    help='iteration interval to perform state backups')
parser.add_argument('--batch-size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--milestones', default='10,20,30', type=str,
                    help='milestones (epochs) for LR decreasing')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--start-epoch', type=int, default=1,
                    help='epoch number to resume')
parser.add_argument('--split', default='train', type=str,
                    help='name of the dataset split used to train')
# Model settings
parser.add_argument('--size', default=400, type=int,
                    help='image size')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

start_epoch = args.start_epoch
image_size = (args.size, args.size)

train_loader = torch.utils.data.DataLoader(
    ImageFilelist(root=args.data_root,
                  flist=args.data_txt,
                  transform=Compose([Scale((args.size, args.size)),
                                     ToTensor()])),
    batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True)

net = resnet.resnet101()
criterion = MultiLLFunction()

if args.cuda:
    net.cuda()
    criterion.cuda()

if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not osp.exists(args.save_folder):
    os.makedirs(args.save_folder)

if osp.exists(args.snapshot):
    net.load_state_dict(torch.load(args.snapshot))

optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = MultiStepLR(
    optimizer, milestones=[int(x) for x in args.milestones.split(',')])
scheduler.step(args.start_epoch)


def train(epoch):
    net.train()
    total_loss = AverageMeter()
    # total_loss = 0
    epoch_loss_stats = AverageMeter()
    # epoch_total_loss = 0
    start_time = time.time()

    for i_batch, sample_batched in enumerate(train_loader):
        im = Variable(sample_batched[0])
        label = Variable(sample_batched[1])
        if args.cuda:
            im = im.cuda()
            label = label.cuda()

        optimizer.zero_grad()
        out_masks = net(im, label)
        out_masks = out_masks.cuda()
        loss = criterion(out_masks, label)
        loss.backward()
        optimizer.step()
        total_loss.update(loss.data[0], im.size(0))
        epoch_loss_stats.update(loss.data[0], im.size(0))

        if i_batch % args.backup_iters == 0:
            filename = 'casenet_{0}_{1}_snapshot.pth'.format(
                args.dataset, args.split)
            filename = osp.join(args.save_folder, filename)
            state_dict = net.state_dict()
            torch.save(state_dict, filename)

            optim_filename = 'casenet_{0}_{1}_optim.pth'.format(
                args.dataset, args.split)
            optim_filename = osp.join(args.save_folder, optim_filename)
            state_dict = optimizer.state_dict()
            torch.save(state_dict, optim_filename)

        if i_batch % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            # cur_loss = total_loss / args.log_interval
            print('[{:5d}] ({:5d}/{:5d}) | ms/batch {:.6f} |'
                  ' loss {:.6f} | lr {:.7f}'.format(
                epoch, i_batch, len(train_loader),
                elapsed_time * 1000, total_loss.avg,
                scheduler.get_lr()[0]))
            total_loss.reset()

        start_time = time.time()

    epoch_total_loss = epoch_loss_stats.avg

    return epoch_total_loss


if __name__ == '__main__':
    print('Beginning training')
    best_val_loss = None
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            scheduler.step()
            train_loss = train(epoch)
            val_loss = train_loss

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| epoch loss {:.6f} |'.format(
                epoch, time.time() - epoch_start_time, train_loss))
            print('-' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
