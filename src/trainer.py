import os
import os.path as osp
import sys
import json
import math

# import glob
# import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as D
from torch.distributions.kl import kl_divergence

import pprint
import matplotlib.pyplot as plt
from comet_ml import Experiment
from tqdm.autonotebook import tqdm

# from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from metrics import calculate_frechet_distance
from sampling_utils import sample_z, sample_view
from model import Generator  # , Discriminator
from space_modules import SceneEncoder
from misc_utils import (
    mkdir_p,
    flatten_json_iterative_solution,
)  # transform_curriculum, new_transform_curriculum
from data_utils import (
    ImageFilelistMultiscale,
    disc_preds_to_label,
    split_images_on_disc,
    show_batch,
)

# from inception import InceptionV3


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Trainer:
    def __init__(self, cfg, log_dir):

        self.log_dir = log_dir
        self.cfg = cfg

        if log_dir:
            self.log = True
            self.model_dir = osp.join(log_dir, "checkpoints")
            mkdir_p(self.model_dir)
            self.logfile = osp.join(log_dir, "logfile.log")
            sys.stdout = Logger(logfile=self.logfile)
            self.summary_writer = SummaryWriter(log_dir)
            # self.summary_writer = tf.summary.create_file_writer(log_dir)
        else:
            self.log = False
            self.summary_writer = None

        if cfg.cuda:
            # s_gpus = cfg.gpu_id.split(",")
            # self.gpus = [int(ix) for ix in s_gpus]
            # self.num_gpus = len(self.gpus)

            self.device = torch.device("cuda")
            # torch.cuda.set_device(self.gpus[0])
            cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        print("Using device", self.device)

        self.generator = Generator(**cfg.model.generator).to(self.device)
        self.scene_encoder = SceneEncoder(**cfg.model.scene_encoder).to(self.device)
        self.presence_prior = D.Bernoulli(cfg.train.scene_encoder.presence_prior)
        self.scale_prior = D.Normal(
            loc=cfg.train.scene_encoder.scale_prior[0],
            scale=cfg.train.scene_encoder.scale_prior[1],
        )
        self.center_prior = D.Normal(
            loc=cfg.train.scene_encoder.center_prior[0],
            scale=cfg.train.scene_encoder.center_prior[1],
        )
        # print("Initiating InceptionNet")
        # block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        # self.inception = InceptionV3([block_idx])
        # self.inception.train(False)

        print("Generator")
        print(self.generator)
        print("Scene Encoder")
        print(self.scene_encoder)

        self.optimizer = optim.Adam(
            [
                {
                    "params": self.generator.parameters(),
                    "lr": cfg.train.generator.optimizer.lr,
                    "betas": cfg.train.generator.optimizer.betas,
                },
                {
                    "params": self.scene_encoder.parameters(),
                    "lr": cfg.train.scene_encoder.optimizer.lr,
                    "betas": cfg.train.scene_encoder.optimizer.betas,
                },
            ],
        )

        # self.bce = nn.BCEWithLogitsLoss().to(self.device)
        self.l1_loss = nn.L1Loss()

        # TODO resume model

        self.comet = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            workspace=os.getenv("COMET_WORKSPACE"),
            project_name=cfg.comet_project_name,
            disabled=cfg.logcomet is False or not self.log,
        )
        self.comet.set_name(f"{cfg.experiment_name}/{cfg.run_name}")
        self.comet.log_parameters(flatten_json_iterative_solution(self.cfg))
        self.comet.log_parameter(
            "CUDA_VISIBLE_DEVICES", getattr(os.environ, "CUDA_VISIBLE_DEVICES", "-1")
        )
        self.comet.log_asset_data(json.dumps(self.cfg, indent=4), file_name="cfg.json")
        self.comet.set_model_graph(f"{self.generator}\n{self.scene_encoder}")
        if cfg.cfg_file:
            self.comet.log_asset(cfg.cfg_file)
        self.comet.log_asset_folder("src")

        self.curr_step = 0
        self.epoch = 0
        if self.log:
            with open(osp.join(self.log_dir, "cfg.json"), "w") as f:
                json.dump(cfg, f, indent=4)

        self.prepare_dataset(self.cfg.train.data_dir)
        self.print_info()

    def prepare_dataset(self, data_dir):
        self.dataset = ImageFilelistMultiscale(
            data_dir, "*.png", ((64, 64), (128, 128))
        )
        self.num_tr = len(self.dataset)
        self.steps_per_epoch = int(math.ceil(self.num_tr / self.cfg.train.batch_size))

    def prepare_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.cuda,
        )

    def print_info(self):
        print("Using config:")
        pprint.pprint(self.cfg)
        print("\n")
        pprint.pprint("Size of train dataset: {}".format(self.num_tr))
        print("\n")

    def recon_loss(self, decoded, target):
        return self.l1_loss(decoded, target)

    def compute_kl_loss(
        self,
        presence_logits,
        valid_indexes,
        scale_mean,
        scale_std,
        center_shift_mean,
        center_shift_std,
        # presence_prior,
        # scale_prior,
        # center_prior,
    ):
        presence_likelihood = D.Bernoulli(logits=presence_logits)
        presence_kl = kl_divergence(presence_likelihood, self.presence_prior).mean()

        scale_kl = center_kl = 0.0
        if valid_indexes.sum() > 0:
            scale_mean = scale_mean[valid_indexes]
            scale_std = scale_std[valid_indexes]
            scale_likelihood = D.Normal(loc=scale_mean, scale=scale_std)
            scale_kl = kl_divergence(scale_likelihood, self.scale_prior).mean()

            center_mean = center_shift_mean[valid_indexes]
            center_std = center_shift_std[valid_indexes]
            center_likelihood = D.Normal(loc=center_mean, scale=center_std)
            center_kl = kl_divergence(center_likelihood, self.center_prior).mean()

        return presence_kl, scale_kl, center_kl

    def train_epoch(self, epoch):
        train_loader = self.prepare_dataloader()
        pbar = tqdm(
            enumerate(train_loader),
            total=self.steps_per_epoch,
            ncols=0,
            desc=f"E:{epoch}",
            mininterval=30,
            miniters=50,
        )
        total_recon_loss = 0.0
        total_presence_kl = 0.0
        total_scale_kl = 0.0
        total_center_kl = 0.0
        total_num_objs = 0
        counter = 1

        self.generator.train(True)
        self.scene_encoder.train(True)
        for it, (image_batch64, image_batch128) in pbar:

            image_batch64 = image_batch64.to(self.device)
            image_batch128 = image_batch128.to(self.device)

            if self.cfg.train.scene_encoder.random_noise or (self.curr_step <= 10000):
                image_batch = image_batch128 + torch.normal(
                    0, 0.02, image_batch128.size(), device=self.device
                )
                image_batch = image_batch.clamp(0, 1)
            encoded_scene = self.scene_encoder(image_batch)
            z_bg = encoded_scene["bg_feats"]
            view_in_bg = encoded_scene["bg_transform_params"]
            z_fg = encoded_scene["valid_img_feats"]
            view_in_fg = encoded_scene["fg_transform_params"]

            reconstructed = self.generator(
                z_bg=z_bg, z_fg=z_fg, view_in_bg=view_in_bg, view_in_fg=view_in_fg
            )
            recon_loss = self.recon_loss(reconstructed, image_batch64)

            valid_indexes = encoded_scene["obj_pres"] >= 0.5
            presence_logits = encoded_scene["glimpses_info"]["pres_p_logits"]
            scale_mean, scale_std = (
                encoded_scene["glimpses_info"]["scale_mean"],
                encoded_scene["glimpses_info"]["scale_std"],
            )
            center_mean, center_std = (
                encoded_scene["glimpses_info"]["center_shift_mean"],
                encoded_scene["glimpses_info"]["center_shift_std"],
            )
            presence_kl, scale_kl, center_kl = self.compute_kl_loss(
                presence_logits,
                valid_indexes,
                scale_mean,
                scale_std,
                center_mean,
                center_std,
            )
            kl_loss = self.cfg.train.scene_encoder.kl_weight * (
                presence_kl + scale_kl + center_kl
            )
            loss = recon_loss + kl_loss

            loss.backward()
            self.optimizer.step()
            self.generator.zero_grad()
            self.scene_encoder.zero_grad()

            total_recon_loss += recon_loss.detach().cpu().numpy()
            total_presence_kl += presence_kl.detach().cpu().numpy()
            total_scale_kl += scale_kl.detach().cpu().numpy()
            total_center_kl += center_kl.detach().cpu().numpy()
            nobjs = z_fg.size(1)
            total_num_objs += nobjs

            pbar.set_postfix(
                recon=f"{recon_loss.detach().cpu().numpy():.3f}({total_recon_loss / (counter):.3f})",
                presence=f"{presence_kl.detach().cpu().numpy():.1E}({total_presence_kl / (counter):.1E})",
                scale=f"{scale_kl.detach().cpu().numpy():.3f}({total_scale_kl / (counter):.3f})",
                center=f"{center_kl.detach().cpu().numpy():.2f}({total_center_kl / (counter):.2f})",
                nobjs=f"{nobjs}({total_num_objs / (counter):.1f})",
                refresh=False,
            )

            if it % (self.cfg.train.it_log_interval) == 0:

                self.log_training(
                    recon_loss=total_recon_loss / counter,
                    presence_kl=total_presence_kl / counter,
                    scale_kl=total_scale_kl / counter,
                    center_kl=total_center_kl / counter,
                    source_images=image_batch64,
                    reconstructed_images=reconstructed,
                    reconstructed_nobjs=valid_indexes.sum((1, 2)),
                    nobjs=total_num_objs / counter,
                    epoch=epoch,
                    it=it,
                )

                total_recon_loss = 0.0
                total_presence_kl = 0.0
                total_scale_kl = 0.0
                total_center_kl = 0.0
                total_num_objs = 0
                counter = 0

            counter += 1

    # def calculate_frechet_distance(self, epoch, it, num_samples=32):
    #     num_samples = min(max(num_samples, 2), len(self.dataset))
    #     real_images = torch.stack(
    #         [self.dataset[i] for i in torch.randint(len(self.dataset), (num_samples,))]
    #     )
    #     z_bg = sample_z(num_samples, self.generator.z_dim_bg, num_objects=1)
    #     z_fg = sample_z(
    #         num_samples,
    #         self.generator.z_dim_fg,
    #         num_objects=torch.randint(3, 11, (1,)).item(),
    #         # num_objects=(
    #         #     3,
    #         #     min(
    #         #         10, 3 + 1 * (epoch // self.cfg.train.obj_num_increase_epoch)
    #         #     ),
    #         # ),
    #     )
    #     bg_view = sample_view(
    #         batch_size=num_samples,
    #         num_objects=1,
    #         azimuth_range=(-10, 10),
    #         elevation_range=(-5, 5),
    #         scale_range=(0.9, 1.1),
    #     )
    #     fg_view = sample_view(
    #         batch_size=num_samples,
    #         num_objects=z_fg.shape[1],
    #         **new_transform_curriculum(epoch),
    #     )
    #     with torch.no_grad():
    #         fake_images = self.generator(
    #             z_bg.to(self.device),
    #             z_fg.to(self.device),
    #             bg_view.to(self.device),
    #             fg_view.to(self.device),
    #         )
    #         torch.cuda.empty_cache()
    #         self.inception = self.inception.to(self.device)
    #         fid = calculate_frechet_distance(
    #             real_images.to(self.device), fake_images, self.inception
    #         )
    #         self.inception = self.inception.cpu()
    #         del real_images, fake_images, z_bg, z_fg, bg_view, fg_view
    #         torch.cuda.empty_cache()

    #     if self.log:
    #         self.summary_writer.add_scalar("fid", fid, global_step=self.curr_step + it)
    #         self.comet.log_metrics(
    #             {"fid": fid}, step=self.curr_step + it, epoch=epoch,
    #         )
    #     return fid

    def log_training(
        self,
        recon_loss,
        presence_kl,
        scale_kl,
        center_kl,
        source_images,
        reconstructed_images,
        reconstructed_nobjs,
        nobjs,
        epoch,
        it,
    ):
        if self.log:
            curr_step = self.curr_step + it
            self.summary_writer.add_scalar(
                "losses/recon_loss", recon_loss, global_step=curr_step,
            )
            self.summary_writer.add_scalar(
                "losses/presence_kl", presence_kl, global_step=curr_step,
            )
            self.summary_writer.add_scalar(
                "losses/scale_kl", scale_kl, global_step=curr_step,
            )
            self.summary_writer.add_scalar(
                "losses/center_kl", center_kl, global_step=curr_step,
            )
            self.summary_writer.add_scalar(
                "num_objs", nobjs, global_step=curr_step,
            )

            self.comet.log_metrics(
                {
                    "recon_loss": recon_loss,
                    "presence_kl": presence_kl,
                    "scale_kl": scale_kl,
                    "center_kl": center_kl,
                    "num_objects": nobjs,
                },
                step=curr_step,
                epoch=epoch,
            )
            fig = plt.figure(figsize=(16, 10))
            plt.imshow(
                T.functional.to_pil_image(
                    make_grid(
                        torch.cat((source_images, reconstructed_images), 0)
                        .detach()
                        .cpu(),
                        nrow=4,
                    )
                )
            )
            plt.title(
                "Top: Input, Bottom: Generated\nPredicted nobjs: "
                + str(reconstructed_nobjs.tolist())
            )
            self.comet.log_figure(
                figure=fig, figure_name="samples", step=curr_step, epoch=epoch,
            )
            plt.close(fig)

    def train(self):
        print("Start training")
        for epoch in range(0, self.cfg.train.epochs):
            self.epoch = epoch
            with self.comet.train():
                self.train_epoch(epoch)
            self.curr_step += self.steps_per_epoch
            # self.start_epoch.assign_add(1)
            # if self.log and (((epoch + 1) % self.cfg.train.snapshot_interval) == 0):
            #     self.ckpt_manager.save(epoch + 1)

    def save_model(self, epoch):
        pass
