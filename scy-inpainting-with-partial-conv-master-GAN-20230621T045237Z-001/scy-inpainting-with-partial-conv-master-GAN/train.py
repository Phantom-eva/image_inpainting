import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from evaluation import evaluate
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from net import discriminator
from places2 import Places2
from util.io import load_ckpt
from util.io import save_ckpt
from write_data import write_excel_xls


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='./srv/datasets/Places2')
parser.add_argument('--mask_root', type=str, default='./masks')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=20000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/images'.format(args.save_dir))
    os.makedirs('{:s}/ckpt'.format(args.save_dir))
    os.makedirs('{:s}/fake'.format(args.save_dir))
    os.makedirs('{:s}/real'.format(args.save_dir))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

size = (args.image_size, args.image_size)
img_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor(),
     transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
mask_tf = transforms.Compose(
    [transforms.Resize(size=size), transforms.ToTensor()])

dataset_train = Places2(args.root, args.mask_root, img_tf, mask_tf, 'train')
dataset_val = Places2(args.root, args.mask_root, img_tf, mask_tf, 'val')

iterator_train = iter(data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads))
print(len(dataset_train))
model = PConvUNet().to(device)
D = discriminator().to(device)
if args.finetune:
    lr = args.lr_finetune
    model.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
D_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, D.parameters()), lr=lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# D_optimizer = torch.optim.Adam(model.parameters(), lr=lr)



BCE_loss = nn.BCELoss().cuda()

criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', model)], [('optimizer', optimizer)])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)

for i in tqdm(range(start_iter, args.max_iter)):
    model.train()
    D.train()

    # Train the discriminator
    #D.zero_grad()
    image, mask, gt = [x.to(device) for x in next(iterator_train)]
    #mask = 1 - mask
    output, _ = model(image, mask)
    #output_comp = mask * image + (1 - mask) * output_D
    label = mask*image
    D_realinput = torch.cat([label, gt], 1)
    D_realresult = D(D_realinput).squeeze()
    D_fakeinput = torch.cat([label, output], 1)
    D_fakeresult = D(D_fakeinput).squeeze()
    D_realloss = BCE_loss(D_realresult, torch.ones(D_realresult.size()).cuda())
    D_fakeloss = BCE_loss(D_fakeresult, torch.zeros(D_fakeresult.size()).cuda())
    D_train_loss = (D_realloss + D_fakeloss) / 2
    D_optimizer.zero_grad()
    D_train_loss.backward(retain_graph=True)


    if (i + 1) % args.log_interval == 0:
        writer.add_scalar('discriminator_loss', D_train_loss.item(), i+1)
        write_excel_xls('./excel_data/discriminator_loss.xls', D_train_loss.item())

    # train generator
    #model.zero_grad()
    #output, _ = model(image, mask)
    #G_result_comp = mask * image + (1 - mask) * output
    loss_dict = criterion(image, mask, output, gt)

    loss = 0.0
    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)
            write_excel_xls('./excel_data/loss_{:s}.xls'.format(key), value.item())

    #D_result_forG = D(D_fakeinput).squeeze()
    G_BCE_loss = BCE_loss(D_fakeresult, torch.ones(D_fakeresult.size()).cuda())
    if (i + 1) % args.log_interval == 0:
        writer.add_scalar('G_BCE_loss', G_BCE_loss.item(), i + 1)
        write_excel_xls('./excel_data/G_BCE_loss.xls', G_BCE_loss.item())
    G_train_loss = loss + G_BCE_loss
    optimizer.zero_grad()
    G_train_loss.backward()
    D_optimizer.step()
    optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                  [('model', model)], [('optimizer', optimizer)], i + 1)

    if (i + 1) % args.vis_interval == 0:
        model.eval()
        evaluate(model, dataset_val, device,
                  '{:s}/images/test_{:d}.jpg'.format(args.save_dir, i + 1),
                  '{:s}/real/real_{:d}.jpg'.format(args.save_dir, i + 1),
                  '{:s}/fake/fake_{:d}.jpg'.format(args.save_dir, i + 1))


writer.close()
