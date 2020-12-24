import argparse
import gc

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from datasets import ImageStainingDataset
from models import GeneratorResNet


CHANNELS = 3
PATCH_SIZE = 768
RESIDUAL_BLOCKS = 9

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,
                    help="Model path of generator", required=True)
args = parser.parse_args()

input_shape = (CHANNELS, PATCH_SIZE, PATCH_SIZE)

G_AB = GeneratorResNet(input_shape, RESIDUAL_BLOCKS)
G_AB = G_AB.cuda()
G_AB.load_state_dict(torch.load(args.model_path))
G_AB.eval()


def fold(unfolded, patch_size, output_size, channels=3):
    fold_params = dict(kernel_size=(
        patch_size, patch_size), stride=(patch_size))

    unfolded = unfolded.contiguous().view(1, 1, -1, channels*patch_size*patch_size)
    unfolded = unfolded.permute(0, 1, 3, 2)
    unfolded = unfolded.view(1, channels*patch_size*patch_size, -1)

    return nn.Fold(output_size=output_size, **fold_params)(unfolded)


def unfold(img, patch_size, channels=3):
    return img.unfold(0, 3, 1).unfold(1, patch_size, patch_size).unfold(
        2, patch_size, patch_size).reshape(-1, 3, patch_size, patch_size)


transform = [
    transforms.ToTensor()
]


dataloader = DataLoader(
    ImageStainingDataset("../data/val", transform=transform),
    batch_size=1,
    num_workers=2,
)


def pad_img(img, padding):
    return nn.functional.pad(img, (padding // 2, padding // 2,
                                   padding // 2, padding // 2))


def crop_center(img, padding):
    return img[:,
               int(padding//2):int(img.size(1)-padding//2),
               int(padding//2):int(img.size(2)-padding//2)
               ]


for i, batch in enumerate(dataloader):
    img_from = batch['from'].squeeze(0)

    assert img_from.size(1) == img_from.size(2)

    padding = PATCH_SIZE - (img_from.size(1) % PATCH_SIZE)

    img_from = pad_img(img_from, padding)
    output_size = (img_from.size()[-1], img_from.size()[-2])

    unfolded = unfold(img_from, PATCH_SIZE)

    unfolded_patches = []

    with torch.no_grad():
        for j, patch_from in enumerate(unfolded.squeeze(0)):
            patch_from = patch_from.unsqueeze(0)
            patch_from = patch_from.cuda()

            generated_patch = G_AB(patch_from)
            unfolded_patches.append(generated_patch)

    unfolded_patches = torch.cat(unfolded_patches, 0)

    folded = fold(unfolded_patches, PATCH_SIZE, output_size)
    folded = crop_center(folded.squeeze(0), padding)

    real_from = torch.cuda.FloatTensor(batch['from'].cuda())
    real_from = make_grid(real_from, nrow=5, normalize=True)

    real_to = torch.cuda.FloatTensor(batch['to'].cuda())
    real_to = make_grid(real_to, nrow=5, normalize=True)

    result = torch.cat((real_from, folded, real_to), 1)
    save_image(result, f"results/{batch['name'][0]}.jpg", normalize=False)
