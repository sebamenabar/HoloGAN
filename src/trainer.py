import os
import os.path as osp
import sys
import json
import math
import glob
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import pprint
import matplotlib.pyplot as plt
from comet_ml import Experiment
from tqdm.autonotebook import tqdm

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter

from metrics import calculate_frechet_distance
from sampling_utils import sample_z, sample_view
from model import Generator, Discriminator
from misc_utils import mkdir_p, flatten_json_iterative_solution, transform_curriculum
from data_utils import (
    ImageFilelist,
    disc_preds_to_label,
    split_images_on_disc,
    show_batch,
)
from inception import InceptionV3



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
        self.discriminator = Discriminator(**cfg.model.discriminator).to(self.device)
        if self.discriminator.style_discriminator:
            return_layers = {
                "style_classifiers.0": "style0",
                "style_classifiers.1": "style1",
                "style_classifiers.2": "style2",
                "style_classifiers.3": "style3",
            }
        else:
            return_layers = {}
        self.d_mid_getter = MidGetter(
            self.discriminator, return_layers=return_layers, keep_output=True
        )
        print('Initiating InceptionNet')
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.inception = InceptionV3([block_idx])

        print("Generator")
        print(self.generator)
        print("Discriminator")
        print(self.discriminator)

        self.g_optimizer = optim.Adam(
            self.generator.parameters(), **cfg.train.generator.optimizer
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), **cfg.train.discriminator.optimizer
        )

        self.bce = nn.BCEWithLogitsLoss().to(self.device)

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
            'CUDA_VISIBLE_DEVICES', getattr(os.environ, 'CUDA_VISIBLE_DEVICES', '-1')
        )
        self.comet.log_asset_data(json.dumps(self.cfg, indent=4), file_name="cfg.json")
        self.comet.set_model_graph(f"{self.generator}\n{self.discriminator}")
        if cfg.cfg_file:
            self.comet.log_asset(cfg.cfg_file)
        self.comet.log_asset_folder("src")

        # self.start_epoch = tf.Variable(0)
        self.curr_step = 0
        self.epoch = 0
        # self.ckpt = tf.train.Checkpoint(
        #     generator=self.generator,
        #     discriminator=self.discriminator,
        #     g_optimizer=self.g_optimizer,
        #     d_optimizer=self.d_optimizer,
        #     start_epoch=self.start_epoch,
        #     curr_step=self.curr_step,
        # )
        # # if cfg.train.resume:
        #     ckpt_resumer = tf.train.CheckpointManager(
        #         self.ckpt, cfg.train.resume, max_to_keep=3,
        #     )
        #     # if a checkpoint exists, restore the latest checkpoint.
        #     if ckpt_resumer.latest_checkpoint:
        #         self.ckpt.restore(ckpt_resumer.latest_checkpoint)
        #         print("Latest checkpoint restored!!", ckpt_resumer.latest_checkpoint)
        #         print(
        #             f"Last epoch trained:{self.start_epoch.numpy()}, Current step: {self.curr_step.numpy()}"
        #         )
        if self.log:
            with open(osp.join(self.log_dir, "cfg.json"), "w") as f:
                json.dump(cfg, f, indent=4)
            # self.ckpt_manager = tf.train.CheckpointManager(
            #     self.ckpt, self.model_dir, max_to_keep=3
            # )

        self.prepare_dataset(self.cfg.train.data_dir)
        self.print_info()

        if self.cfg.train.generator.fixed_z:
            self.z_bg = sample_z(1, self.generator.z_dim_bg, num_objects=1).to(
                self.device
            )
            self.z_fg = sample_z(1, self.generator.z_dim_fg, num_objects=1).to(
                self.device
            )
            self.bg_view = sample_view(1, num_objects=1).to(self.device)
            self.fg_view = sample_view(1, num_objects=1).to(self.device)
        else:
            self.z_bg = self.z_fg = self.bg_view = self.fg_view = None

    def prepare_dataset(self, data_dir):
        self.dataset = ImageFilelist(data_dir, "*.png")
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

    # lossess
    def discriminator_loss(self, real, generated, style_real=None, style_fake=None):
        real_loss = self.bce(real, torch.ones_like(real))
        generated_loss = self.bce(generated, torch.zeros_like(generated))
        total_disc_loss = real_loss + generated_loss

        if style_real:
            for logits in style_real:
                style_real_loss = self.bce(logits, torch.ones_like(logits))
                total_disc_loss = total_disc_loss + style_real_loss
        if style_fake:
            for logits in style_fake:
                style_fake_loss = self.bce(logits, torch.zeros_like(logits))
                total_disc_loss = total_disc_loss + style_fake_loss

        return total_disc_loss

    def generator_loss(self, generated, style_fake=None):
        total_g_loss = self.bce(generated, torch.ones_like(generated))
        if style_fake:
            for logits in style_fake:
                style_g_loss = self.bce(logits, torch.ones_like(logits))
                total_g_loss = total_g_loss + style_g_loss

        return total_g_loss

    def batch_logits(self, image_batch, z_bg, z_fg, bg_view, fg_view):
        generated = self.generator(z_bg, z_fg, bg_view, fg_view)

        d_fake_logits = self.discriminator(generated)
        # image_batch = image_batch * 2) - 1
        if self.cfg.train.discriminator.random_noise or self.curr_step <= 2000:
            image_batch = image_batch + torch.normal(0, 0.1, image_batch.size())
        d_real_logits = self.discriminator(image_batch)

        return d_fake_logits, d_real_logits, generated

    def train_epoch(self, epoch):
        if epoch == 0:
            fid = self.calculate_frechet_distance(epoch, 0)
            print(f'RANDOM FID: {fid:.3f}')
        train_iter = self.prepare_dataloader()
        pbar = tqdm(
            enumerate(train_iter),
            total=self.steps_per_epoch,
            ncols=0,
            desc=f"E:{epoch}",
            mininterval=10,
            miniters=50,
        )
        total_d_loss = 0.0
        total_g_loss = 0.0
        counter = 1
        real_are_real_samples_counter = 0
        real_samples_counter = 0
        fake_are_fake_samples_counter = 0
        fake_samples_counter = 0
        l1 = 0.0

        z_bg = z_fg = None
        self.generator.train(True)
        self.discriminator.train(True)
        for it, image_batch in pbar:
            bsz = image_batch.shape[0]
            # generated random noise
            if self.z_bg is not None:
                # For overfitting one sample and debugging
                z_bg = self.z_bg.repeat(bsz, 1, 1)
                z_fg = self.z_fg.repeat(bsz, 1, 1)
                bg_view = self.bg_view
                fg_view = self.fg_view
            else:
                z_bg = sample_z(bsz, self.generator.z_dim_bg, num_objects=1)
                z_fg = sample_z(
                    bsz,
                    self.generator.z_dim_fg,
                    num_objects=torch.randint(3, 11, (1,)).item(),
                )
                bg_view = sample_view(
                    batch_size=bsz,
                    num_objects=1,
                    azimuth_range=(-3, 3),
                    elevation_range=(-3, 3),
                    scale_range=(0.95, 1.05),
                )
                fg_view = sample_view(
                    batch_size=bsz,
                    num_objects=z_fg.shape[1],
                    **transform_curriculum(epoch - 5),
                )

            image_batch = image_batch.to(self.device)
            z_bg = z_bg.to(self.device)
            z_fg = z_fg.to(self.device)
            bg_view = bg_view.to(self.device)
            fg_view = fg_view.to(self.device)

            generated = self.generator(z_bg, z_fg, bg_view, fg_view)
            d_fake_style_logits, d_fake_logits = self.d_mid_getter(generated.detach())
<<<<<<< HEAD
            # image_batch = (image_batch * 2) - 1
            if self.cfg.train.discriminator.random_noise or (self.curr_step <= 10000):
=======
            if self.cfg.train.discriminator.random_noise or self.curr_step <= 10000:
>>>>>>> 8ca03edc84474a53807d7f82d866e3572734ef03
                image_batch = image_batch + torch.normal(
                    0, 0.02, image_batch.size(), device=self.device
                )
            d_real_style_logits, d_real_logits = self.d_mid_getter(image_batch)

            d_loss = self.discriminator_loss(
                d_real_logits,
                d_fake_logits,
                d_real_style_logits.values(),
                d_fake_style_logits.values(),
            )
            self.discriminator.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            d_fake_style_logits, d_fake_logits = self.d_mid_getter(generated)
            g_loss = self.generator_loss(d_fake_logits, d_fake_style_logits.values())
            self.generator.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            total_d_loss += d_loss.detach().cpu().numpy()
            # total_g_loss += g_loss.numpy() / self.cfg.train.generator.update_freq
            total_g_loss += g_loss.detach().cpu().numpy()

            real_samples_counter += d_real_logits.size(0)
            fake_samples_counter += d_fake_logits.size(0)

            real_are_real = (d_real_logits >= 0).cpu().numpy().sum()
            real_are_real_samples_counter += real_are_real

            fake_are_fake = (d_fake_logits < 0).cpu().numpy().sum()
            fake_are_fake_samples_counter += fake_are_fake

            # according to paper generator makes 2 steps per each step of the disc
            # for _ in range(self.cfg.train.generator.update_freq - 1):
            #     with tf.GradientTape(persistent=True) as tape:
            #         # fake img
            #         d_fake_logits, _, generated = self.batch_logits(
            #             image_batch, z_bg, z_fg
            #         )
            #         g_loss = self.generator_loss(d_fake_logits)
            #     g_variables = self.generator.trainable_variables
            #     g_gradients = tape.gradient(g_loss, g_variables)
            #     self.g_optimizer.apply_gradients(zip(g_gradients, g_variables))
            #     total_g_loss += g_loss.numpy() / self.cfg.train.generator.update_freq

            pbar.set_postfix(
                g_loss=f"{g_loss.detach().cpu().numpy():.1f}({total_g_loss / (counter):.1f})",
                d_loss=f"{d_loss.detach().cpu().numpy():.3f}({total_d_loss / (counter):.3f})",
                rrr=f"{real_are_real / d_real_logits.shape[0]:.1f}({real_are_real_samples_counter / real_samples_counter:.1f})",
                frf=f"{fake_are_fake / d_fake_logits.shape[0]:.1f}({fake_are_fake_samples_counter / fake_samples_counter:.1f})",
                l1=l1,
                refresh=False,
            )

            if it % (self.cfg.train.it_log_interval) == 0:
                with torch.no_grad():
                    z_fg_pad = torch.zeros(z_fg.size(0), 1, z_fg.size(2)).to(
                        z_fg.device
                    )
                    z_fg_padded = torch.cat((z_fg_pad, z_fg), 1)
                    view_fg_pad = sample_view(bsz, num_objects=1).to(fg_view.device)
                    view_fg_padded = torch.cat((view_fg_pad, fg_view), 1)

                    voxel = self.generator.gen_voxel_only(z_bg, z_fg, bg_view, fg_view)
                    pad_voxel = self.generator.gen_voxel_only(
                        z_bg, z_fg_padded, bg_view, view_fg_padded
                    )
                    l1 = (voxel - pad_voxel).abs().mean().item()

                    del (
                        z_fg_pad,
                        z_fg_padded,
                        view_fg_pad,
                        view_fg_padded,
                        voxel,
                        pad_voxel,
                    )

                self.log_training(
                    d_loss=total_d_loss / counter,
                    g_loss=total_g_loss / counter,
                    real_are_real=real_are_real_samples_counter / real_samples_counter,
                    fake_are_fake=fake_are_fake_samples_counter / fake_samples_counter,
                    fake_images=generated,
                    real_images=image_batch,
                    d_fake_logits=d_fake_logits,
                    d_real_logits=d_real_logits,
                    epoch=epoch,
                    it=it,
                    l1=l1,
                )
                real_are_real_samples_counter = 0
                fake_are_fake_samples_counter = 0
                real_samples_counter = 0
                fake_samples_counter = 0
                total_d_loss = 0.0
                total_g_loss = 0.0
                counter = 0

            counter += 1
            gc.collect()

        del train_iter
        fid = self.calculate_frechet_distance(epoch, it)
        print(f'EPOCH {epoch} FID: {fid:.3f}')

    def calculate_frechet_distance(self, epoch, it, num_samples=64):
        num_samples = min(max(num_samples, 2), len(self.dataset))
        real_images = torch.stack(
            [self.dataset[i] for i in torch.randint(len(self.dataset), (num_samples,))]
        )
        z_bg = sample_z(num_samples, self.generator.z_dim_bg, num_objects=1)
        z_fg = sample_z(
            num_samples,
            self.generator.z_dim_fg,
            num_objects=torch.randint(3, 11, (1,)).item(),
            # num_objects=(
            #     3,
            #     min(
            #         10, 3 + 1 * (epoch // self.cfg.train.obj_num_increase_epoch)
            #     ),
            # ),
        )
        bg_view = sample_view(
            batch_size=num_samples,
            num_objects=1,
            azimuth_range=(-3, 3),
            elevation_range=(-3, 3),
            scale_range=(0.95, 1.05),
        )
        fg_view = sample_view(
            batch_size=num_samples,
            num_objects=z_fg.shape[1],
            **transform_curriculum(epoch - 5),
        )
        with torch.no_grad():
            fake_images = self.generator.cpu()(z_bg, z_fg, bg_view, fg_view)
            fid = calculate_frechet_distance(real_images, fake_images, self.inception)
        if self.log:
            self.summary_writer.add_scalar('fid', fid, global_step=self.curr_step + it)
            self.comet.log_metrics(
                {
                    "fid": fid,
                },
                step=self.curr_step + it,
                epoch=epoch,
            )
        return fid

    def log_training(
        self,
        d_loss,
        g_loss,
        fake_images,
        real_images,
        d_fake_logits,
        d_real_logits,
        epoch,
        it,
        real_are_real,
        fake_are_fake,
        l1,
    ):
        if self.log:
            curr_step = self.curr_step + it
            real_are_real_images, real_are_fake_images = split_images_on_disc(
                real_images, d_real_logits
            )
            fake_are_real_images, fake_are_fake_images = split_images_on_disc(
                fake_images, d_fake_logits
            )
            # with self.summary_writer.as_default():
            self.summary_writer.add_scalar(
                "losses/d_loss",
                d_loss,
                global_step=curr_step,
                # description="Average of predicting real images as real and fake as fake",
            )
            self.summary_writer.add_scalar(
                "losses/g_loss",
                g_loss,
                global_step=curr_step,
                # description="Predicting fake images as real",
            )
            self.summary_writer.add_scalar(
                "l1_padded_voxel",
                l1,
                global_step=curr_step,
                # description="Predicting fake images as real",
            )
            self.summary_writer.add_scalar(
                "accuracy/real",
                real_are_real,
                global_step=curr_step,
                # description="Real images classified as real",
            )
            self.summary_writer.add_scalar(
                "accuracy/fake",
                fake_are_fake,
                global_step=curr_step,
                # description="Fake images classified as fake",
            )
            if fake_are_fake_images.size(0) > 0:
                self.summary_writer.add_image(
                    f"fake/are_fake",
                    make_grid(fake_are_fake_images),
                    # max_outputs=25,
                    global_step=curr_step,
                    # description="Fake images that the discriminator says are fake",
                )
            if fake_are_real_images.size(0) > 0:
                self.summary_writer.add_image(
                    f"fake/are_real",
                    make_grid(fake_are_real_images),
                    # max_outputs=25,
                    global_step=curr_step,
                    # description="Fake images that the discriminator says are real",
                )
            if real_are_fake_images.size(0) > 0:
                self.summary_writer.add_image(
                    f"real/are_fake",
                    make_grid(real_are_fake_images),
                    # max_outputs=25,
                    global_step=curr_step,
                    # description="Real images that the discriminator says are fake",
                )
            if real_are_real_images.size(0) > 0:
                self.summary_writer.add_image(
                    f"real/are_real",
                    make_grid(real_are_real_images),
                    # max_outputs=25,
                    global_step=curr_step,
                    # description="Real images that the discriminator says are real",
                )

            self.comet.log_metrics(
                {
                    "d_loss": d_loss,
                    "g_loss": g_loss,
                    "accuracy/real": real_are_real,
                    "accuracy/fake": fake_are_fake,
                    "l1_padded_voxel": l1,
                },
                step=curr_step,
                epoch=epoch,
            )
            fig = show_batch(fake_images, labels=disc_preds_to_label(d_fake_logits))
            self.comet.log_figure(
                figure=fig, figure_name="" f"fake_{epoch}_{it}.jpg", step=curr_step,
            )
            plt.close(fig)
            fig = show_batch(real_images, labels=disc_preds_to_label(d_real_logits))
            self.comet.log_figure(
                figure=fig, figure_name="" f"real_{epoch}_{it}.jpg", step=curr_step,
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
