import argparse
import os
import numpy as np
import time
import datetime
import sys
from PIL import Image
import random

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from datasets import ImageStainingDataset
from models import Discriminator, GeneratorUNet, weights_init_normal


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0,
                    help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200,
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
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=1024,
                    help="size of image height")
parser.add_argument("--img_width", type=int, default=1024,
                    help="size of image width")
parser.add_argument("--channels", type=int, default=3,
                    help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=5, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int,
                    default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

#  Calculate output of image discriminator (PatchGAN)
output_discriminator = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load(
        "saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load(
        "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


def unfold(img):
    patch_size = 1024
    return img.unfold(0, 3, 1).unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size).reshape(-1, 3, patch_size, patch_size)


transform = [
    transforms.ToTensor(),
    transforms.Lambda(lambda img: unfold(img))
]


dataloader = DataLoader(
    ImageStainingDataset("../data/train", transform=transform),
    batch_size=opt.batch_size,
    shuffle=True,
)

# # Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    batch = next(iter(dataloader))
    real_from = batch["from"]
    real_to = batch["to"]

    i = random.randint(0, len(real_from[0]) - 1)

    real_from = real_from[0][i].unsqueeze(0)
    real_to = real_to[0][i].unsqueeze(0)

    real_from = Tensor(real_from.cuda())
    real_to = Tensor(real_to.cuda())
    fake_to = generator(real_from)
    img_sample = torch.cat((real_from.data, fake_to.data, real_to.data), -2)

    output_name = f"{batches_done}_{batch['name'][0]}"

    save_image(img_sample, "images/%s/%s.png" %
               (opt.dataset_name, output_name), nrow=5, normalize=True)


prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        for j, (patches_from, patches_to) in enumerate(zip(batch["from"], batch["to"])):
            patches_split_from = patches_from.split(3)
            patches_split_to = patches_to.split(3)

            for k, (patch_split_from, patch_split_to) in enumerate(zip(patches_split_from, patches_split_to)):
                # Model inputs
                real_from = Tensor(patch_split_from.cuda())
                real_to = Tensor(patch_split_to.cuda())

                # Adversarial ground truths
                valid = Tensor(
                    np.ones((real_from.size(0), *output_discriminator)))
                fake = Tensor(
                    np.zeros((real_from.size(0), *output_discriminator)))

                # ------------------
                #  Train Generators
                # ------------------

                optimizer_G.zero_grad()

                # GAN loss
                fake_to = generator(real_from)
                pred_fake = discriminator(fake_to, real_from)
                loss_GAN = criterion_GAN(pred_fake, valid)
                # Pixel-wise loss
                loss_pixel = criterion_pixelwise(fake_to, real_to)

                # Total loss
                loss_G = loss_GAN + lambda_pixel * loss_pixel

                loss_G.backward()

                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Real loss
                pred_real = discriminator(real_to, real_from)
                loss_real = criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = discriminator(real_to.detach(), real_from)
                loss_fake = criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                optimizer_D.step()

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
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)
