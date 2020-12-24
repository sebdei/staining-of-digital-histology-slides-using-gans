import glob
import os
import numpy as np
import re
import math

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageStainingDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transforms.Compose(transform)

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        from_index = index * 2
        to_index = from_index + 1

        file_from = self.files[from_index]
        file_to = self.files[to_index]

        name_from = re.search(".*\/(.*)(?:.he|.ihc)", file_from).group(1)
        name_to = re.search(".*\/(.*)(?:.he|.ihc)", file_to).group(1)

        assert name_from == name_to

        img_from = Image.open(file_from)
        img_to = Image.open(file_to)

        # if np.random.random() < 0.5:
        #     img_from = Image.fromarray(np.array(img_from)[:, ::-1, :], "RGB")
        #     img_to = Image.fromarray(np.array(img_to)[:, ::-1, :], "RGB")

        img_from = self.transform(img_from)
        img_to = self.transform(img_to)

        label_from = re.search(
            "\d*-\d*-\w*-\w*-\w*-(.*)\.(he|ihc)\.*", file_from).group(1)
        label_to = re.search(
            "\d*-\d*-\w*-\w*-\w*-(.*)\.(he|ihc)\.*", file_to).group(1)

        if (label_from != label_to):
            raise "Labels are not identical!"

        return {
            "from": img_from,
            "to": img_to,
            "name": name_from,
            "label": label_from
        }

    def __len__(self):
        count = len(self.files)
        if len(self.files) % 2 != 0:
            raise "Files count is not even"
        return int(count / 2)
