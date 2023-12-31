import argparse
import torch
from torchvision import transforms
import os
import opt
from places2 import Places2
from evaluation import evaluate_test
from net import PConvUNet
from util.io import load_ckpt

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./srv/datasets/Places2')
parser.add_argument('--snapshot', type=str, default='./snapshots/default/ckpt/300000.pth')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--mask_root', type=str, default='./masks')
args = parser.parse_args()

device = torch.device('cuda')

size = (args.image_size, args.image_size)
img_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_transform = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_val = Places2(args.root, args.mask_root, img_transform, mask_transform, 'val')

model = PConvUNet().to(device)
load_ckpt(args.snapshot, [('model', model)])

model.eval()
evaluate_test(model, dataset_val, device, 'real', 'fake', 'img')
