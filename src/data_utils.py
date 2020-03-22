import os
import glob
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def split_images_on_disc(images, disc_logits):
    if len(disc_logits.shape) == 2:
        disc_logits = torch.squeeze(disc_logits, 1)
    are_real = disc_logits >= 0.5
    return images[are_real], images[~are_real]


def disc_preds_to_label(disc_logits):
    if len(disc_logits.shape) == 2:
        disc_logits = torch.squeeze(disc_logits, 1)
    disc_p = torch.sigmoid(disc_logits)
    labels = np.where((disc_p >= 0.5).cpu().numpy(), "Real", "False")
    return labels


def show_batch(image_batch, labels=None):
    if (image_batch < 0).numpy().any():
        image_batch = (image_batch + 1) / 2
    fig = plt.figure(figsize=(10, 10))
    for n in range(min(25, image_batch.shape[0])):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        try:
            plt.title(labels[n])
        except:
            pass
        plt.axis("off")
    return fig


def default_loader(path):
    return Image.open(path).convert("RGB")


default_transform = transforms.Compose(
    (transforms.Resize((64, 64),), transforms.ToTensor(),)
)


class ImageFilelist(data.Dataset):
    def __init__(
        self, imgs_dir, match, transform=default_transform, loader=default_loader
    ):
        self.transform = transform
        self.loader = loader
        self.imgs_dir = imgs_dir
        self.files = glob.glob(os.path.join(imgs_dir, match))

        if transform is None:
            self.transform = transforms.functional.to_tensor

    def __getitem__(self, index):
        img_fp = self.files[index]
        img = self.loader(img_fp)
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)
