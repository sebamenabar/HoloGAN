import os
import os.path as osp
import sys
import json
import math
import glob
import gc

import tensorflow as tf
from tensorflow.keras import optimizers, losses

import pprint
import matplotlib.pyplot as plt
from comet_ml import Experiment
from tqdm.autonotebook import tqdm

from sampling_utils import sample_z, sample_view
from model import Generator, Discriminator
from misc_utils import mkdir_p, flatten_json_iterative_solution
from data_utils import (
    process_path,
    prepare_for_training,
    show_batch,
    disc_preds_to_label,
    split_images_on_disc,
)


AUTOTUNE = tf.data.experimental.AUTOTUNE


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
            self.summary_writer = tf.summary.create_file_writer(log_dir)
        else:
            self.log = False

        self.generator = Generator(**cfg.model.generator)
        self.discriminator = Discriminator(**cfg.model.discriminator)

        self.g_optimizer = optimizers.Adam(**cfg.train.generator.optimizer)
        self.d_optimizer = optimizers.Adam(**cfg.train.discriminator.optimizer)

        self.bce = losses.BinaryCrossentropy(from_logits=True)

        # TODO resume model

        self.comet = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            workspace=os.getenv("COMET_WORKSPACE"),
            project_name=cfg.comet_project_name,
            disabled=cfg.logcomet is False or not self.log,
        )
        self.comet.set_name(f"{cfg.experiment_name}/{cfg.run_name}")
        self.comet.log_parameters(flatten_json_iterative_solution(self.cfg))
        self.comet.log_asset_data(json.dumps(self.cfg, indent=4), file_name="cfg.json")
        if cfg.cfg_file:
            self.comet.log_asset(cfg.cfg_file)

        self.start_epoch = tf.Variable(0)
        self.curr_step = tf.Variable(0)
        self.ckpt = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            g_optimizer=self.g_optimizer,
            d_optimizer=self.d_optimizer,
            start_epoch=self.start_epoch,
            curr_step=self.curr_step,
        )
        if cfg.train.resume:
            ckpt_resumer = tf.train.CheckpointManager(
                self.ckpt, cfg.train.resume, max_to_keep=3,
            )
            # if a checkpoint exists, restore the latest checkpoint.
            if ckpt_resumer.latest_checkpoint:
                self.ckpt.restore(ckpt_resumer.latest_checkpoint)
                print("Latest checkpoint restored!!", ckpt_resumer.latest_checkpoint)
                print(
                    f"Last epoch trained:{self.start_epoch.numpy()}, Current step: {self.curr_step.numpy()}"
                )
        if self.log:
            with open(osp.join(self.log_dir, "cfg.json"), "w") as f:
                json.dump(cfg, f, indent=4)
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, self.model_dir, max_to_keep=3
            )

        self.prepare_dataset(self.cfg.train.data_dir)
        self.print_info()

        if self.cfg.train.generator.fixed_z:
            self.z_bg = sample_z(1, self.generator.z_dim_bg, num_objects=1)
            self.z_fg = sample_z(1, self.generator.z_dim_fg, num_objects=1)
            self.bg_view = sample_view(1, num_objects=1)
            self.fg_view = sample_view(1, num_objects=1)
        else:
            self.z_bg = self.z_fg = self.bg_view = self.fg_view = None

    def prepare_dataset(self, data_dir):
        self.data_dir = data_dir
        self.num_tr = len(glob.glob(osp.join(self.data_dir, "*.png")))
        self.list_ds_train = tf.data.Dataset.list_files(
            os.path.join(self.data_dir, "*.png")
        )
        self.labeled_ds = self.list_ds_train.map(
            lambda x: process_path(
                x, self.cfg.train.image_height, self.cfg.train.image_width
            ),
            num_parallel_calls=AUTOTUNE,
        )
        self.steps_per_epoch = int(math.ceil(self.num_tr / self.cfg.train.batch_size))

    def print_info(self):
        print("Using config:")
        pprint.pprint(self.cfg)
        print("\n")
        pprint.pprint("Size of train dataset: {}".format(self.num_tr))
        print("\n")

    # lossess
    def discriminator_loss(self, real, generated):
        real_loss = self.bce(tf.ones_like(real), real)
        generated_loss = self.bce(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.bce(tf.ones_like(generated), generated)

    # def generate_random_noise(self, batch_size, num_objects=(3, 10)):
    #     z_bg = tf.random.uniform(
    #         (batch_size, self.generator.z_dim_bg), minval=-1, maxval=1
    #     )
    #     num_objs = tf.random.uniform(
    #         (batch_size,),
    #         minval=num_objects[0],
    #         maxval=num_objects[1] + 1,
    #         dtype=tf.int32,
    #     )
    #     tensors = []
    #     max_len = max(num_objs)
    #     for no in num_objs:
    #         _t = tf.random.uniform((no, self.generator.z_dim_fg), minval=-1, maxval=1)
    #         _z = tf.zeros((max_len - no, self.generator.z_dim_fg), dtype=tf.float32)
    #         _t = tf.concat((_t, _z), axis=0)
    #         tensors.append(_t)
    #     z_fg = tf.stack(tensors, axis=0)

    #     return z_bg, z_fg

    def batch_logits(self, image_batch, z_bg, z_fg, bg_view, fg_view):
        generated = self.generator(z_bg, z_fg, bg_view, fg_view)

        d_fake_logits = self.discriminator(generated, training=True)
        image_batch = (image_batch * 2) - 1
        if self.cfg.train.discriminator.random_noise or self.curr_step <= 2000:
            image_batch = image_batch + tf.random.normal(image_batch.shape, stddev=0.01)
        d_real_logits = self.discriminator(image_batch, training=True,)

        return d_fake_logits, d_real_logits, generated

    # @tf.function
    def train_epoch(self, epoch):
        train_iter = prepare_for_training(
            self.labeled_ds, self.cfg.train.batch_size, cache=False,
        )
        pbar = tqdm(
            enumerate(train_iter),
            total=self.steps_per_epoch,
            ncols=20,
            desc=f"Epoch {epoch}",
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

        z_bg = z_fg = None
        for it, image_batch in pbar:
            bsz = image_batch.shape[0]
            # generated random noise
            if self.z_bg is not None:
                # For overfitting one sample and debugging
                z_bg = tf.repeat(self.z_bg, bsz, axis=0)
                z_fg = tf.repeat(self.z_fg, bsz, axis=0)
                bg_view = self.bg_view
                fg_view = self.fg_view
            else:
                z_bg = sample_z(bsz, self.generator.z_dim_bg, num_objects=1)
                z_fg = sample_z(
                    bsz,
                    self.generator.z_dim_fg,
                    num_objects=(3, min(10, 3 + 1 * (epoch // 2))),
                )
                bg_view = sample_view(
                    batch_size=bsz,
                    num_objects=1,
                    azimuth_range=(-20, 20),
                    elevation_range=(-10, 10),
                    scale_range=(0.9, 1.1),
                )
                fg_view = sample_view(batch_size=bsz, num_objects=z_fg.shape[1])

            with tf.GradientTape(persistent=True) as tape:
                # fake img
                d_fake_logits, d_real_logits, generated = self.batch_logits(
                    image_batch, z_bg, z_fg, bg_view, fg_view
                )
                d_loss = self.discriminator_loss(d_real_logits, d_fake_logits)
                g_loss = self.generator_loss(d_fake_logits)

            total_d_loss += d_loss.numpy()
            # total_g_loss += g_loss.numpy() / self.cfg.train.generator.update_freq
            total_g_loss += g_loss.numpy()

            d_variables = self.discriminator.trainable_variables
            d_gradients = tape.gradient(d_loss, d_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, d_variables))

            g_variables = self.generator.trainable_variables
            g_gradients = tape.gradient(g_loss, g_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, g_variables))

            del tape

            real_samples_counter += d_real_logits.shape[0]
            fake_samples_counter += d_fake_logits.shape[0]

            real_are_real = (d_real_logits >= 0).numpy().sum()
            real_are_real_samples_counter += real_are_real

            fake_are_fake = (d_fake_logits < 0).numpy().sum()
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
                g_loss=f"{g_loss.numpy():.2f} ({total_g_loss / (counter):.2f})",
                d_loss=f"{d_loss.numpy():.4f} ({total_d_loss / (counter):.4f})",
                rrr=f"{real_are_real / d_real_logits.shape[0]:.1f} ({real_are_real_samples_counter / real_samples_counter:.1f})",
                frf=f"{fake_are_fake / d_fake_logits.shape[0]:.1f} ({fake_are_fake_samples_counter / fake_samples_counter:.1f})",
                refresh=False,
            )

            if it % (self.cfg.train.it_log_interval) == 0:
                self.log_training(
                    d_loss=total_d_loss / counter,
                    g_loss=total_g_loss / counter,
                    real_are_real=real_are_real_samples_counter / real_samples_counter,
                    fake_are_fake=fake_are_fake_samples_counter / fake_samples_counter,
                    fake_images=(generated + 1) / 2,
                    real_images=image_batch,
                    d_fake_logits=d_fake_logits,
                    d_real_logits=d_real_logits,
                    epoch=epoch,
                    it=it,
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
    ):
        if self.log:
            curr_step = (self.curr_step + it).numpy()
            real_are_real_images, real_are_fake_images = split_images_on_disc(
                real_images, d_real_logits
            )
            fake_are_real_images, fake_are_fake_images = split_images_on_disc(
                fake_images, d_fake_logits
            )
            with self.summary_writer.as_default():
                tf.summary.scalar(
                    "losses/d_loss",
                    d_loss,
                    step=curr_step,
                    description="Average of predicting real images as real and fake as fake",
                )
                tf.summary.scalar(
                    "losses/g_loss",
                    g_loss,
                    step=curr_step,
                    description="Predicting fake images as real",
                )
                tf.summary.scalar(
                    "accuracy/real",
                    real_are_real,
                    step=curr_step,
                    description="Real images classified as real",
                )
                tf.summary.scalar(
                    "accuracy/fake",
                    fake_are_fake,
                    step=curr_step,
                    description="Fake images classified as fake",
                )
                tf.summary.image(
                    f"{epoch}-{curr_step}-fake/are_fake",
                    fake_are_fake_images,
                    max_outputs=25,
                    step=curr_step,
                    description="Fake images that the discriminator says are fake",
                )
                tf.summary.image(
                    f"{epoch}-{curr_step}-fake/are_real",
                    fake_are_real_images,
                    max_outputs=25,
                    step=curr_step,
                    description="Fake images that the discriminator says are real",
                )
                tf.summary.image(
                    f"{epoch}-{curr_step}-real/are_fake",
                    real_are_fake_images,
                    max_outputs=25,
                    step=curr_step,
                    description="Real images that the discriminator says are fake",
                )
                tf.summary.image(
                    f"{epoch}-{curr_step}-real/are_real",
                    real_are_real_images,
                    max_outputs=25,
                    step=curr_step,
                    description="Real images that the discriminator says are real",
                )

            self.comet.log_metrics(
                {"d_loss": d_loss, "g_loss": g_loss}, step=curr_step, epoch=epoch
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
        for epoch in range(self.start_epoch.numpy(), self.cfg.train.epochs):
            with self.comet.train():
                self.train_epoch(epoch)
            self.curr_step.assign_add(self.steps_per_epoch)
            self.start_epoch.assign_add(1)
            if self.log and (((epoch + 1) % self.cfg.train.snapshot_interval) == 0):
                self.ckpt_manager.save(epoch + 1)

    def save_model(self, epoch):
        pass

