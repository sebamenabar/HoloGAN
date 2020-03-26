import os
import os.path as osp
import argparse
import math

import torch
from torch.nn import optim
from torch.nn.functional import binary_cross_entropy_with_logits as bce, l1_loss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torchnet.meter as meter


import tqdm.autonotebook as tqdm
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from config import cfg, cfg_from_file
from data_utils import ImageFilelist, split_images_on_disc
from model import Generator, Discriminator
from misc_utils import mkdir_p
from sampling_utils import sample_view, sample_z


def build_dataset(data_dir):
    return ImageFilelist(data_dir, "*.png")


def build_dataloader(dataset, batch_size=64, num_workers=0, pin_memory=False):
    return (
        torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        int(math.ceil(len(dataset) / batch_size)),
    )


def set_logdir(experiment_name, run_name):
    logdir = osp.join("manual_training", experiment_name, run_name)
    mkdir_p(logdir)
    ckpt_dir = osp.join(logdir, "checkpoints")
    mkdir_p(ckpt_dir)
    print("Saving output to: {}".format(logdir))
    writer = SummaryWriter(logdir)
    return writer, logdir, ckpt_dir


def build_models(gen_cfg, disc_cfg):
    generator = Generator(**gen_cfg)
    discriminator = Discriminator(**disc_cfg)
    if discriminator.style_discriminator:
        return_layers = {
            "style_classifiers.0": "style0",
            "style_classifiers.1": "style1",
            "style_classifiers.2": "style2",
            "style_classifiers.3": "style3",
        }
    else:
        return_layers = {}
    wrapped_disc = MidGetter(
        discriminator, return_layers=return_layers, keep_output=True
    )

    return generator, wrapped_disc


def style_loss(logits_list, target="ones"):
    loss = 0.0
    if target == "ones":
        target = torch.ones_like(logits_list[0])
    elif target == "zeros":
        target = torch.zeros_like(logits_list[0])
    for logit in logits_list:
        loss = loss + bce(logit, target)
    return loss


def discriminator_loss(real, fake, style_real=None, style_fake=None):
    real_loss = bce(real, torch.ones_like(real))
    fake_loss = bce(fake, torch.zeros_like(fake))
    total_disc_loss = real_loss + fake_loss

    if style_real:
        total_disc_loss = total_disc_loss + style_loss(style_real, "ones")
    if style_fake:
        total_disc_loss = total_disc_loss + style_loss(style_fake, "zeros")

    return total_disc_loss


def generator_loss(generated, style_fake=None):
    total_g_loss = bce(generated, torch.ones_like(generated))
    if style_fake:
        total_g_loss = total_g_loss + style_loss(style_fake, "ones")

    return total_g_loss


def calculate_voxel_l1_loss(generator, voxel, grad_enabled=True):
    voxel = voxel.detach()
    device = voxel.device
    bsz = voxel.size(0)
    z_fg_dim = generator.z_dim_fg
    with torch.set_grad_enabled(grad_enabled):
        z_fg_pad = torch.zeros(bsz, 1, z_fg_dim).to(device)
        view_fg_pad = sample_view(bsz, num_objects=1).to(device)
        # voxel = self.generator.gen_voxel_only(z_bg, z_fg, bg_view, fg_view)
        padded_voxel = generator.fg_generator(z_fg_pad, view_fg_pad)
        padded_voxel = torch.max(torch.cat((voxel.unsqueeze(1), padded_voxel), 1), 1)[0]
        loss = l1_loss(padded_voxel)
    return loss


def train_epoch(
    epoch,
    dataset,
    g,
    d,
    g_opt,
    d_opt,
    train_info,
    device=None,
    disc_noise=False,
    writer=None,
):
    if device is None:
        device = g.device
    loader, steps_per_epoch = build_dataloader(dataset, batch_size=64, pin_memory=True)
    pbar = tqdm(
        enumerate(loader),
        total=steps_per_epoch,
        ncols=0,
        desc=f"E:{epoch}",
        mininterval=5,
        miniters=50,
    )
    meters = {
        "realmAP": meter.AverageValueMeter(),
        "fakemAP": meter.AverageValueMeter(),
        "d_loss": meter.AverageValueMeter(),
        "g_loss": meter.AverageValueMeter(),
        "l1_loss": meter.AverageValueMeter(),
    }

    z_dim_fg = g.z_dim_fg
    z_dim_bg = g.z_dim_bg

    g.zero_grad()
    d.zero_grad()
    g.train(True)
    d.train(True)
    for it, real_images in pbar:
        # prepare data
        bsz = real_images.size(0)
        max_num_objs = torch.randint(3, 11, (1,)).item()
        z_bg = sample_z(bsz, z_dim_bg, num_objects=1)
        z_fg = sample_z(bsz, z_dim_fg, num_objects=max_num_objs)
        bg_view = sample_view(
            batch_size=bsz, num_objects=1, **train_info["bg_view_range"],
        )
        fg_view = sample_view(
            batch_size=bsz, num_objects=z_fg.size(1), **train_info["fg_view_range"]
        )
        real_images = real_images.to(device)
        z_bg = z_bg.to(device)
        z_fg = z_fg.to(device)
        bg_view = bg_view.to(device)
        fg_view = fg_view.to(device)

        fake_images, voxel = g(z_bg, z_fg, bg_view, fg_view, return_voxel=True)
        voxel = voxel.detach()
        l1_loss = calculate_voxel_l1_loss(g, voxel, grad_enable=True)
        if l1_loss.requires_grad:
            l1_loss.backward() * 1e-3
        l1_loss = l1_loss.item()  # to free memory (?)

        if disc_noise:
            real_images = real_images + torch.normal(
                0, 0.02, real_images.size(), device=device
            )
            real_images = real_images.clamp(0, 1)
        d_real_style_logits, d_real_logits = d(real_images)
        d_fake_style_logits, d_fake_logits = d(fake_images.detach())
        d_loss = discriminator_loss(
            d_real_logits,
            d_fake_logits,
            d_real_style_logits.values(),
            d_fake_style_logits.values(),
        )
        d_loss.backward()
        d_optimizer.step()
        d.zero_grad()
        d_loss = d_loss.item()

        d_fake_style_logits, d_fake_logits = d(fake_images)
        g_loss = generator_loss(d_fake_logits, d_fake_style_logits.values())
        # self.generator.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        g.zero_grad()
        g_loss = g_loss.item()

        meters["d_loss"].add(d_loss)
        meters["g_loss"].add(g_loss)
        meters["l1_loss"].add(l1_loss)
        real_acc = (d_real_logits >= 0).cpu().numpy().sum() / bsz
        fake_acc = (d_fake_logits < 0).cpu().numpy().sum() / bsz
        meters["realmAP"].add(real_acc)
        meters["fakemAP"].add(fake_acc)

        pbar.set_postfix(
            g_l=f"{g_loss:.1f}({meters['g_loss'].value():.1f})",
            d_l=f"{d_loss:.2f}({meters['d_loss'].value():.2f})",
            l1_l=f"{l1_loss:.2f}({meters['l1_loss'].value():.2f})",
            rrr=f"{real_acc:.2f}({meters['realmAP'].value():.2f})",
            frf=f"{fake_acc:.2f}({meters['fakemAP'].value():.2f})",
            refresh=False,
        )

        if (it % 200) == 0:
            log_training(
                meters=meters,
                fake_images=fake_images,
                real_images=real_images,
                fake_logits=d_fake_logits,
                real_logits=d_real_logits,
                epoch=epoch,
                curr_step=it + train_info["curr_step"],
            )
            for m in meters.values():
                m.reset()
    return steps_per_epoch


def log_training(
    writer,
    meters,
    real_images,
    fake_images,
    real_logits,
    fake_logits,
    epoch,
    curr_step,
):
    real_are_real_images, real_are_fake_images = split_images_on_disc(
        real_images, real_logits,
    )
    fake_are_real_images, fake_are_fake_images = split_images_on_disc(
        fake_images, fake_logits,
    )
    writer.add_scalar(
        "losses/d_loss",
        meters["d_loss"].value(),
        global_step=curr_step,
        # description="Average of predicting real images as real and fake as fake",
    )
    writer.add_scalar(
        "losses/g_loss",
        meters["g_loss"].value(),
        global_step=curr_step,
        # description="Predicting fake images as real",
    )
    writer.add_scalar(
        "l1_padded_voxel",
        meters["l1_loss"].value(),
        global_step=curr_step,
        # description="Predicting fake images as real",
    )
    writer.add_scalar(
        "accuracy/real",
        meters["realmAP"].value(),
        global_step=curr_step,
        # description="Real images classified as real",
    )
    writer.add_scalar(
        "accuracy/fake",
        meters["fakemAP"].value(),
        global_step=curr_step,
        # description="Fake images classified as fake",
    )
    if fake_are_fake_images.size(0) > 0:
        writer.add_image(
            f"fake/are_fake",
            make_grid(fake_are_fake_images),
            # max_outputs=25,
            global_step=curr_step,
            # description="Fake images that the discriminator says are fake",
        )
    if fake_are_real_images.size(0) > 0:
        writer.add_image(
            f"fake/are_real",
            make_grid(fake_are_real_images),
            # max_outputs=25,
            global_step=curr_step,
            # description="Fake images that the discriminator says are real",
        )
    if real_are_fake_images.size(0) > 0:
        writer.add_image(
            f"real/are_fake",
            make_grid(real_are_fake_images),
            # max_outputs=25,
            global_step=curr_step,
            # description="Real images that the discriminator says are fake",
        )
    if real_are_real_images.size(0) > 0:
        writer.add_image(
            f"real/are_real",
            make_grid(real_are_real_images),
            # max_outputs=25,
            global_step=curr_step,
            # description="Real images that the discriminator says are real",
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", dest="cfg_file", help="optional config file", default=None, type=str
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--gpu-id", dest="gpu_id", default="0")
    # parser.add_argument("--logdir", type=str, default='manual_training')

    parser.add_argument("--disc-noise", type=bool, default=False)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--experiment-name", type=str, required=True)

    args = parser.parse_args()
    if args.cfg_file is not None:
        cfg.cfg_file = args.cfg_file
        cfg_from_file(args.cfg_file)
    return args


def resume_training(ckpt_fp, generator, discriminator, g_optimizer, d_optimizer):
    ckpt = torch.load(ckpt_fp)
    generator.load_state_dict(ckpt["g_state_dict"])
    discriminator.load_state_dict(ckpt["d_state_dict"])
    g_optimizer.load_state_dict(ckpt["g_opt_state_dict"])
    d_optimizer.load_state_dict(ckpt["d_opt_state_dict"])

    return ckpt["train_info"]


FOREGROUND_VIEW_RANGE = dict(
    azimuth_range=(-10, 10), elevation_range=(-5, 5), scale_range=(0.9, 1.1),
)
BACKGROUND_VIEW_RANGE = {
    "azimuth_range": (-5, 5),
    "scale_range": (0.9, 1.1),
    "tx_range": (-0.5, 0.5),
    "tz_range": (-0.5, 0.5),
}

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda")
    dataset = build_dataset()
    writer, logdir, ckpt_dir = set_logdir(args.experiment_name, args.run_name)

    generator, wrapped_disc = build_models(cfg.model.generator, cfg.model.discriminator)
    generator = generator.to(device)
    wrapped_disc = wrapped_disc.to(device)

    g_optimizer = optim.Adam(generator.parameters(), **cfg.train.generator.optimizer)
    d_optimizer = optim.Adam(
        wrapped_disc._model.parameters(), **cfg.train.discriminator.optimizer
    )

    if args.resume:
        train_info = resume_training(
            args.resume, generator, wrapped_disc._model, g_optimizer, d_optimizer,
        )
        train_info["start_epoch"] = train_info["epoch"] + 1
    else:
        train_info = {
            "curr_epoch": 0,
            "curr_step": 0,
            "start_epoch": 0,
        }
    train_info["fg_view_range"] = FOREGROUND_VIEW_RANGE
    train_info["bg_view_range"] = BACKGROUND_VIEW_RANGE

    print("TRAINING WITH")
    print(train_info)

    for epoch in range(
        train_info["start_epoch"], train_info["start_epoch"] + args.epochs
    ):
        train_info["curr_epoch"] = epoch
        steps = train_epoch(
            epoch,
            dataset,
            generator,
            wrapped_disc,
            g_optimizer,
            d_optimizer,
            train_info,
            device,
            args.disc_noise,
            writer,
        )
        train_info["curr_step"] += steps
        torch.save({
            'g': generator.state_dict(),
            'd': wrapped_disc._model.state_dict(),
            'g_opt': g_optimizer.state_dict(),
            'd_opt': d_optimizer.state_dict(),
            'train_info': train_info
        }, osp.join(ckpt_dir, "epoch-{epoch}.pth"))
