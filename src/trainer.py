import os
import os.path as osp
import sys
import json
import math
import glob

import tensorflow as tf
from tensorflow.keras import optimizers, losses

import pprint
from comet_ml import Experiment
from tqdm.autonotebook import tqdm

from model import Generator, Discriminator
from misc_utils import mkdir_p, flatten_json_iterative_solution
from data_utils import process_path, prepare_for_training, show_batch


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

        if self.log:
            with open(osp.join(self.log_dir, "cfg.json"), "w") as f:
                json.dump(cfg, f, indent=4)

            self.ckpt = tf.train.Checkpoint(
                generator=self.generator,
                discriminator=self.discriminator,
                g_optimizer=self.g_optimizer,
                d_optimizer=self.d_optimizer,
            )

            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, self.model_dir, max_to_keep=3
            )
            # if a checkpoint exists, restore the latest checkpoint.
            if self.ckpt_manager.latest_checkpoint:
                self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print("Latest checkpoint restored!!",)

        self.prepare_dataset(self.cfg.train.data_dir)
        self.print_info()

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

    def generate_random_noise(self, batch_size, num_objects=(3, 10)):
        z_bg = tf.random.uniform(
            (batch_size, self.generator.z_dim_bg), minval=-1, maxval=1
        )
        num_objs = tf.random.uniform(
            (batch_size,),
            minval=num_objects[0],
            maxval=num_objects[1] + 1,
            dtype=tf.int32,
        )
        tensors = []
        max_len = max(num_objs)
        for no in num_objs:
            _t = tf.random.uniform((no, self.generator.z_dim_fg), minval=-1, maxval=1)
            _z = tf.zeros((max_len - no, self.generator.z_dim_fg), dtype=tf.float32)
            _t = tf.concat((_t, _z), axis=0)
            tensors.append(_t)
        z_fg = tf.stack(tensors, axis=0)

        return z_bg, z_fg

    def batch_logits(self, image_batch, z_bg, z_fg):
        generated = self.generator(z_bg, z_fg)

        d_fake_logits = self.discriminator(generated, training=True)
        image_batch = (image_batch * 2) - 1
        if self.cfg.train.discriminator.random_noise:
            image_batch = image_batch + tf.random.normal(image_batch.shape, stddev=0.02)
        d_real_logits = self.discriminator(image_batch, training=True,)

        return d_fake_logits, d_real_logits, generated

    # @tf.function
    def train_epoch(self, epoch):
        train_iter = prepare_for_training(
            self.labeled_ds, self.cfg.train.batch_size, cache=True
        )
        pbar = tqdm(
            enumerate(train_iter),
            total=self.steps_per_epoch,
            ncols=20,
            desc=f"Epoch {epoch}",
        )
        total_d_loss = 0.0
        total_g_loss = 0.0
        counter = 1
        for it, image_batch in pbar:
            bsz = image_batch.shape[0]
            # generated random noise
            z_bg, z_fg = self.generate_random_noise(bsz, (3, min(10, 3 + 1 * epoch)))
            with tf.GradientTape(persistent=True) as tape:
                # fake img
                d_fake_logits, d_real_logits, _ = self.batch_logits(
                    image_batch, z_bg, z_fg
                )
                d_loss = self.discriminator_loss(d_real_logits, d_fake_logits)
                g_loss = self.generator_loss(d_fake_logits)

            total_d_loss += d_loss.numpy()
            total_g_loss += g_loss.numpy() / self.cfg.train.generator.update_freq

            d_variables = self.discriminator.trainable_variables
            d_gradients = tape.gradient(d_loss, d_variables)
            self.d_optimizer.apply_gradients(zip(d_gradients, d_variables))

            g_variables = self.generator.trainable_variables
            g_gradients = tape.gradient(g_loss, g_variables)
            self.g_optimizer.apply_gradients(zip(g_gradients, g_variables))

            # according to paper generator makes 2 steps per each step of the disc
            for _ in range(self.cfg.train.generator.update_freq - 1):
                with tf.GradientTape(persistent=True) as tape:
                    # fake img
                    d_fake_logits, _, generated = self.batch_logits(
                        image_batch, z_bg, z_fg
                    )
                    g_loss = self.generator_loss(d_fake_logits)
                g_variables = self.generator.trainable_variables
                g_gradients = tape.gradient(g_loss, g_variables)
                self.g_optimizer.apply_gradients(zip(g_gradients, g_variables))
                total_g_loss += g_loss.numpy() / self.cfg.train.generator.update_freq

            pbar.set_postfix(
                g_loss=f"{g_loss.numpy():.4f} ({total_g_loss / (counter):.4f})",
                d_loss=f"{d_loss.numpy():.4f} ({total_d_loss / (counter):.4f})",
            )
            pbar.refresh()

            if it % (self.cfg.train.it_log_interval) == 0:
                self.log_training(
                    d_loss=total_d_loss / counter,
                    g_loss=total_g_loss / counter,
                    generated_images=(generated + 1) / 2,
                    epoch=epoch,
                    it=it,
                )
                total_d_loss = 0.0
                total_g_loss = 0.0
                counter = 0

            counter += 1

    def log_training(self, d_loss, g_loss, generated_images, epoch, it):
        if self.log:
            curr_step = epoch * self.steps_per_epoch + it

            with self.summary_writer.as_default():
                tf.summary.scalar("losses/d_loss", d_loss, step=curr_step)
                tf.summary.scalar("losses/g_loss", g_loss, step=curr_step)
                tf.summary.image(
                    f"generated_{epoch}_{it}.jpg",
                    generated_images,
                    max_outputs=25,
                    step=curr_step,
                )

            self.comet.log_metrics(
                {"d_loss": d_loss, "g_loss": g_loss}, step=curr_step, epoch=epoch
            )
            fig = show_batch(generated_images)
            self.comet.log_figure(
                figure=fig,
                figure_name="" f"generated_{epoch}_{it}.jpg",
                step=curr_step,
            )
            fig.close()

    def train(self):
        print("Start training")
        for epoch in range(self.cfg.train.epochs):
            with self.comet.train():
                self.train_epoch(epoch)
            if self.log and ((epoch + 1) % self.cfg.train.snapshot_interval == 0):
                self.ckpt_manager.save(epoch)

    def save_model(self, epoch):
        pass

