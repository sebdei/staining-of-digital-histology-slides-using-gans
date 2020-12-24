import argparse
import os
import numpy as np
import itertools
import sys
import datetime
import time
import random
from PIL import Image


import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader

from models import GeneratorResNet, Discriminator, weights_init_normal
from datasets import ImageStainingDataset
from utils import ReplayBuffer, LambdaLR

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500,
                    help="number of epochs of training")
parser.add_argument("--dataset_name", type=str,
                    default="staining", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100,
                    help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=2,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=412,
                    help="size of image")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50,
                    help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10,
                    help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9,
                    help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float,
                    default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0,
                    help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.MSELoss()
criterion_identity = torch.nn.MSELoss()

cuda = torch.cuda.is_available()

print(f'runs on cuda: {cuda}')

input_shape = (opt.channels, opt.img_size, opt.img_size)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load(
        "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load(
        "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" %
                                   (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" %
                                   (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(
    D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(
    D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_from_buffer = ReplayBuffer()
fake_to_buffer = ReplayBuffer()


# Image transformations
transform = [
    transforms.RandomCrop((opt.img_size, opt.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]


dataloader = DataLoader(
    ImageStainingDataset("../data/train", transform=transform),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    ImageStainingDataset("../data/val", transform=transform),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)


def sample_images(batches_done):
    G_AB.eval()
    G_BA.eval()

    batch = next(iter(val_dataloader))
    real_from = batch["from"]
    real_to = batch["to"]

    real_from = Tensor(real_from.cuda())
    real_to = Tensor(real_to.cuda())

    fake_to = G_AB(real_from)
    fake_from = G_BA(real_to)

    # Arange images along x-axis
    real_from = make_grid(real_from, nrow=5, normalize=True)
    real_to = make_grid(real_to, nrow=5, normalize=True)
    fake_from = make_grid(fake_from, nrow=5, normalize=True)
    fake_to = make_grid(fake_to, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_from, fake_to, real_to, fake_from), 1)
    save_image(image_grid, "images/%s/%s.jpg" %
               (opt.dataset_name, batches_done), normalize=False)


prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_from = Tensor(batch['from'].cuda())
        real_to = Tensor(batch['to'].cuda())

        # Adversarial ground truths
        valid = Tensor(
            np.ones((real_from.size(0), *D_A.output_shape)))
        fake = Tensor(
            np.zeros((real_from.size(0), *D_A.output_shape)))

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_from), real_from)
        loss_id_B = criterion_identity(G_AB(real_to), real_to)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_to = G_AB(real_from)
        loss_GAN_AB = criterion_GAN(D_B(fake_to), valid)
        fake_from = G_BA(real_to)
        loss_GAN_BA = criterion_GAN(D_A(fake_from), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_to)
        loss_cycle_A = criterion_cycle(recov_A, real_from)
        recov_B = G_AB(fake_from)
        loss_cycle_B = criterion_cycle(recov_B, real_to)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_from), valid)
        # Fake loss (on batch of previously generated samples)
        fake_from_ = fake_from_buffer.push_and_pop(fake_from)
        loss_fake = criterion_GAN(D_A(fake_from_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_to), valid)
        # Fake loss (on batch of previously generated samples)
        fake_to_ = fake_to_buffer.push_and_pop(fake_to)
        loss_fake = criterion_GAN(D_B(fake_to_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" %
                   (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" %
                   (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" %
                   (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" %
                   (opt.dataset_name, epoch))
