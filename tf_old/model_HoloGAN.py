from __future__ import division
import os
import sys
from glob import glob
import json
import shutil
from datetime import datetime as dt


with open(sys.argv[1], 'r') as fh:
    cfg=json.load(fh)
IMAGE_PATH       = cfg['image_path']
OUTPUT_DIR = cfg['output_dir']
LOGDIR = os.path.join(OUTPUT_DIR, "log")

from imageio import imwrite as imsave
from tools.ops import *
from tools.utils import get_image, merge, inverse_transform, to_bool
from tools.rotation_utils import *
from tools.model_utils import transform_voxel_to_match_image


#----------------------------------------------------------------------------

class HoloGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=False,
         output_height=128, output_width=128,
         gf_dim=64, df_dim=64,
         c_dim=3, dataset_name='clevr',
         input_fname_pattern='*.webp'):

    self.sess = sess
    self.crop = crop

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.gf_dim = gf_dim
    self.df_dim = df_dim
    self.c_dim = c_dim

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
    self.checkpoint_dir = LOGDIR

  def build(self, build_func_name):
      build_func = None
      if build_func_name == 'build_HoloGAN':
          build_func = self.build_HoloGAN
      build_func()

  def build_HoloGAN(self):
    # view_in_bg: Background transformation params [bsz, (azimuth y-rot, elevation x-rot, z-rot, scale, tx, ty, tx)]
    self.view_in_bg = tf.placeholder(tf.float32, [None, 6], name='view_in_bg')
    # view_in_fg: Foreground transform params [bsz, num_obects, params(6)]
    self.view_in_fg = tf.placeholder(tf.float32, [None, 6], name='view_in_fg')
    # inputs: for real images
    self.inputs = tf.placeholder(tf.float32, [None, self.output_height, self.output_width, self.c_dim], name='real_images')
    # z_bg: 
    self.z_bg = tf.placeholder(tf.float32, [None, None, cfg['z_dim_bg']], name='z_bg')
    # z_fg:
    self.z_fg = tf.placeholder(tf.float32, [None, None, cfg['z_dim_fg']], name='z_fg')

    inputs = self.inputs

    funcs = {
        'generator_AdaIN': self.generator_AdaIN,
        'generator_AdaIN_block': self.generator_AdaIN_block,
        'generator_AdaIN_res128': self.generator_AdaIN_res128,
        'discriminator_IN': self.discriminator_IN,
        'discriminator_IN_style_res128': self.discriminator_IN_style_res128,
        'train_HoloGAN': self.train_HoloGAN,
        'generate_random_rotation_translation': generate_random_rotation_translation,
    }
    # gen_func = eval("self." + (cfg['generator']))
    # dis_func = eval("self." + (cfg['discriminator']))
    # self.gen_view_func = eval(cfg['view_func'])
    gen_func = funcs[cfg['generator']]
    dis_func = funcs[cfg['discriminator']]
    self.gen_view_func = funcs[cfg['view_func']]

    self.G = gen_func(self.z_bg, self.z_fg, self.view_in_bg, self.view_in_fg)

    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style Disc")
        # self.D, self.D_logits, _, self.d_h1_r, self.d_h2_r, self.d_h3_r, self.d_h4_r = dis_func(inputs, cont_dim=cfg['z_dim'], reuse=False)
        self.D, self.D_logits = dis_func(inputs, cont_dim=cfg['z_dim_fg'], reuse=False)
        # self.D_, self.D_logits_, self.Q_c_given_x, self.d_h1_f, self.d_h2_f, self.d_h3_f, self.d_h4_f = dis_func(self.G, cont_dim=cfg['z_dim'], reuse=True)
        self.D_, self.D_logits_ = dis_func(self.G, cont_dim=cfg['z_dim_fg'], reuse=True)

        # self.d_h1_loss = cfg["DStyle_lambda"] * (
        #             tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_r, tf.ones_like(self.d_h1_r))) \
        #             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h1_f, tf.zeros_like(self.d_h1_f))))
        # self.d_h2_loss = cfg["DStyle_lambda"] * (
        #             tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_r, tf.ones_like(self.d_h2_r))) \
        #             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h2_f, tf.zeros_like(self.d_h2_f))))
        # self.d_h3_loss = cfg["DStyle_lambda"] * (
        #             tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_r, tf.ones_like(self.d_h3_r))) \
        #             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h3_f, tf.zeros_like(self.d_h3_f))))
        # self.d_h4_loss = cfg["DStyle_lambda"] * (
        #             tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_r, tf.ones_like(self.d_h4_r))) \
        #             + tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.d_h4_f, tf.zeros_like(self.d_h4_f))))
    else:
        # self.D, self.D_logits, _ = dis_func(inputs, cont_dim=cfg['z_dim_fg'], reuse=False)
        # self.D_, self.D_logits_, self.Q_c_given_x = dis_func(self.G, cont_dim=cfg['z_dim_fg'], reuse=True)
        self.D, self.D_logits = dis_func(inputs, cont_dim=cfg['z_dim_fg'], reuse=False)
        self.D_, self.D_logits_ = dis_func(self.G, cont_dim=cfg['z_dim_fg'], reuse=True)


    self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))


    if str.lower(str(cfg["style_disc"])) == "true":
        print("Style disc")
        self.d_loss = self.d_loss + self.d_h1_loss + self.d_h2_loss + self.d_h3_loss + self.d_h4_loss
    #====================================================================================================================
    #Identity loss

    ## TODO fix identity loss
    # self.q_loss = cfg["lambda_latent"] * tf.reduce_mean(tf.square(self.Q_c_given_x - self.z))
    # self.d_loss = self.d_loss + self.q_loss
    ## TEMP why is g_loss
    # self.g_loss = self.g_loss + self.q_loss


    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train_HoloGAN(self, config):
      self.d_lr_in = tf.placeholder(tf.float32, None, name='d_eta')
      self.g_lr_in = tf.placeholder(tf.float32, None, name='d_eta')

      d_optim = tf.train.AdamOptimizer(cfg['d_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.d_loss, var_list=self.d_vars)
      g_optim = tf.train.AdamOptimizer(cfg['g_eta'], beta1=cfg['beta1'], beta2=cfg['beta2']).minimize(self.g_loss, var_list=self.g_vars)

      tf.global_variables_initializer().run()

      shutil.copyfile(sys.argv[1], os.path.join(LOGDIR, 'config.json'))
      self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
      self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
      self.writer = SummaryWriter(os.path.join(LOGDIR, str(dt.now())), self.sess.graph)

      num_objs = 1
      # Sample noise Z and view parameters to test during training
      sample_z_bg = self.sampling_Z(cfg['z_dim_bg'], num_objs=1, type=str(cfg['sample_z']))
      sample_z_fg = self.sampling_Z(cfg['z_dim_fg'], num_objs=num_objs, type=str(cfg['sample_z']))
      sample_view_bg = self.gen_view_func(cfg['batch_size'] * 1,
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=to_bool(str(cfg['with_translation'])),
                                       with_scale=to_bool(str(cfg['with_scale'])))
      sample_view_fg = self.gen_view_func(cfg['batch_size'] * num_objs,
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=to_bool(str(cfg['with_translation'])),
                                       with_scale=to_bool(str(cfg['with_scale'])))
      # sample_view_bg = tf.reshape(sample_view_bg, (cfg['batch_size'], 1, *sample_view_bg.shape[1:]))
      # sample_view_fg = tf.reshape(sample_view_fg, (cfg['batch_size'], num_objs, *sample_view_fg.shape[1:]))
      sample_files = self.data[0:cfg['batch_size']]

      if config.dataset == "cats" or config.dataset == "cars":
          sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for sample_file in sample_files]
      else:
          sample_images = [get_image(sample_file,
                                    input_height=self.input_height,
                                    input_width=self.input_width,
                                    resize_height=self.output_height,
                                    resize_width=self.output_width,
                                    crop=self.crop) for sample_file in sample_files]

      ## TEMP
      batch_images = sample_images
      # batch_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
      # batch_view = self.gen_view_func(cfg['batch_size'],
      #                           cfg['ele_low'], cfg['ele_high'],
      #                           cfg['azi_low'], cfg['azi_high'],
      #                           cfg['scale_low'], cfg['scale_high'],
      #                           cfg['x_low'], cfg['x_high'],
      #                           cfg['y_low'], cfg['y_high'],
      #                           cfg['z_low'], cfg['z_high'],
      #                           with_translation=False,
      #                           with_scale=to_bool(str(cfg['with_translation'])))

      batch_z_bg = self.sampling_Z(cfg['z_dim_bg'], num_objs=1, type=str(cfg['sample_z']))
      batch_z_fg = self.sampling_Z(cfg['z_dim_fg'], num_objs=num_objs, type=str(cfg['sample_z']))
      batch_view_bg = self.gen_view_func(cfg['batch_size'] * 1,
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=to_bool(str(cfg['with_translation'])),
                                       with_scale=to_bool(str(cfg['with_scale'])))
      batch_view_fg = self.gen_view_func(cfg['batch_size'] * num_objs,
                                       cfg['ele_low'], cfg['ele_high'],
                                       cfg['azi_low'], cfg['azi_high'],
                                       cfg['scale_low'], cfg['scale_high'],
                                       cfg['x_low'], cfg['x_high'],
                                       cfg['y_low'], cfg['y_high'],
                                       cfg['z_low'], cfg['z_high'],
                                       with_translation=to_bool(str(cfg['with_translation'])),
                                       with_scale=to_bool(str(cfg['with_scale'])))
      # batch_view_bg = tf.reshape(batch_view_bg, (cfg['batch_size'], 1, *batch_view_bg.shape[1:]))
      # batch_view_fg = tf.reshape(batch_view_fg, (cfg['batch_size'], num_objs, *batch_view_fg.shape[1:]))

      counter = 0
      start_time = time.time()
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")

      self.data = glob.glob(os.path.join(IMAGE_PATH, self.input_fname_pattern))
      d_lr = cfg['d_eta']
      g_lr = cfg['g_eta']
      for epoch in range(cfg['max_epochs']):
          d_lr = d_lr if epoch < cfg['epoch_step'] else d_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])
          g_lr = g_lr if epoch < cfg['epoch_step'] else g_lr * (cfg['max_epochs'] - epoch) / (cfg['max_epochs'] - cfg['epoch_step'])

          random.shuffle(self.data)
          batch_idxs = min(len(self.data), config.train_size) // cfg['batch_size']

          for idx in range(0, batch_idxs):
              batch_files = self.data[idx * cfg['batch_size']:(idx + 1) * cfg['batch_size']]
              # if config.dataset == "cats" or config.dataset == "cars":
              #     batch_images = [get_image(batch_file,
              #                       input_height=self.input_height,
              #                       input_width=self.input_width,
              #                       resize_height=self.output_height,
              #                       resize_width=self.output_width,
              #                       crop=self.crop) for batch_file in batch_files]
              # else:
              #     batch_images = [get_image(batch_file,
              #                       input_height=self.input_height,
              #                       input_width=self.input_width,
              #                       resize_height=self.output_height,
              #                       resize_width=self.output_width,
              #                       crop=self.crop) for batch_file in batch_files]

              # batch_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
              # batch_view = self.gen_view_func(cfg['batch_size'],
              #                          cfg['ele_low'], cfg['ele_high'],
              #                          cfg['azi_low'], cfg['azi_high'],
              #                          cfg['scale_low'], cfg['scale_high'],
              #                          cfg['x_low'], cfg['x_high'],
              #                          cfg['y_low'], cfg['y_high'],
              #                          cfg['z_low'], cfg['z_high'],
              #                          with_translation=False,
              #                          with_scale=to_bool(str(cfg['with_translation'])))

              feed = {self.inputs: batch_images,
                      self.z_bg: batch_z_bg,
                      self.z_fg: batch_z_fg,
                      # self.z: batch_z,
                      self.view_in_bg: batch_view_bg,
                      self.view_in_fg: batch_view_fg,
                      self.d_lr_in: d_lr,
                      self.g_lr_in: g_lr}
              # Update D network
              _, summary_str = self.sess.run([d_optim, self.d_sum],feed_dict=feed)
              self.writer.add_summary(summary_str, counter)
              # Update G network
              _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=feed)
              self.writer.add_summary(summary_str, counter)
              # Run g_optim twice
              _, summary_str = self.sess.run([g_optim, self.g_sum],  feed_dict=feed)
              self.writer.add_summary(summary_str, counter)

              errD_fake = self.d_loss_fake.eval(feed)
              errD_real = self.d_loss_real.eval(feed)
              errG = self.g_loss.eval(feed)
              # errQ = self.q_loss.eval(feed)
              errQ = 0.

              print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, q_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                       time.time() - start_time, errD_fake + errD_real, errG, errQ))

              if np.mod(counter, 100) == 0:
                  # self.save(LOGDIR, counter)
                  feed_eval = {self.inputs: sample_images,
                               self.z_bg: sample_z_bg,
                               self.z_fg: sample_z_fg,
                               # self.z: batch_z,
                               self.view_in_bg: sample_view_bg,
                               self.view_in_fg: sample_view_fg,
                               self.d_lr_in: d_lr,
                               self.g_lr_in: g_lr}

                  samples, d_loss, g_loss = self.sess.run(
                      [self.G, self.d_loss, self.g_loss],
                      feed_dict=feed_eval)


                  print('generated samples')
                  print(samples)

                  ren_img = inverse_transform(samples)
                  ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
                  # real_img = inverse_transform(np.array(sample_images))
                  # real_img = np.clip(255 * real_img, 0, 255).astype(np.uint8)

                  try:
                      self.writer.add_summary(tf.summary.image("{0}_GAN.png".format(counter), ren_img).eval(), counter)
                      imsave(
                          os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)),
                          merge(ren_img, [cfg['batch_size'] // 4, 4]))
                      # imsave(
                      #     os.path.join(OUTPUT_DIR, "{0}_reals_GAN.png".format(counter)),
                      #     merge(real_img, [cfg['batch_size'] // 4, 4]))
                      # print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                  except:
                      self.writer.add_summary(tf.summary.image("{0}_GAN.png".format(counter), ren_img[:1]).eval(), counter)
                      imsave(
                          os.path.join(OUTPUT_DIR, "{0}_GAN.png".format(counter)),
                          ren_img[0])
                      # imsave(
                      #     os.path.join(OUTPUT_DIR, "{0}_reals_GAN.png".format(counter)),
                      #     real_img[0])
                      # print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
              counter += 1
          return
      self.save(LOGDIR, counter)

  def sample_HoloGAN(self, config):
      could_load, checkpoint_counter = self.load(self.checkpoint_dir)
      if could_load:
          counter = checkpoint_counter
          print(" [*] Load SUCCESS")
      else:
          print(" [!] Load failed...")
          return
      SAMPLE_DIR = os.path.join(OUTPUT_DIR, "samples")
      if not os.path.exists(SAMPLE_DIR):
          os.makedirs(SAMPLE_DIR)
      sample_z = self.sampling_Z(cfg['z_dim'], str(cfg['sample_z']))
      if config.rotate_azimuth:
          low  = cfg['azi_low']
          high = cfg['azi_high']
          step = 10
      elif config.rotate_elevation:
          low  = cfg['ele_low']
          high = cfg['ele_high']
          step = 5
      else:
          low  = 0
          high = 10
          step = 1

      for i in range(low, high, step):
          if config.rotate_azimuth:
              sample_view = np.tile(
                  np.array([i * math.pi / 180.0, 0 * math.pi / 180.0, 1.0, 0, 0, 0]), (cfg['batch_size'], 1))
          elif config.rotate_azimuth:
              sample_view = np.tile(
                  np.array([270 * math.pi / 180.0, (90 - i) * math.pi / 180.0, 1.0, 0, 0, 0]), (cfg['batch_size'], 1))
          else:
              sample_view = self.gen_view_func(cfg['batch_size'],
                                               cfg['ele_low'], cfg['ele_high'],
                                               cfg['azi_low'], cfg['azi_high'],
                                               cfg['scale_low'], cfg['scale_high'],
                                               cfg['x_low'], cfg['x_high'],
                                               cfg['y_low'], cfg['y_high'],
                                               cfg['z_low'], cfg['z_high'],
                                               with_translation=False,
                                               with_scale=to_bool(str(cfg['with_translation'])))

          feed_eval = {self.z: sample_z,
                       self.view_in: sample_view}

          samples = self.sess.run(self.G, feed_dict=feed_eval)
          ren_img = inverse_transform(samples)
          ren_img = np.clip(255 * ren_img, 0, 255).astype(np.uint8)
          try:
              scipy.misc.imsave(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)),
                  merge(ren_img, [cfg['batch_size'] // 4, 4]))
          except:
              scipy.misc.imsave(
                  os.path.join(SAMPLE_DIR, "{0}_samples_{1}.png".format(counter, i)),
                  ren_img[0])

#=======================================================================================================================

  def sampling_Z(self, z_dim, num_objs=None, type="uniform"):
      if num_objs:
        out_shape = (cfg['batch_size'], num_objs, z_dim)
      else:
        out_shape = (cfg['batch_size'], z_dim)
      # return np.ones(out_shape)
      if str.lower(type) == "uniform":
          return np.random.uniform(-1., 1., out_shape)
      else:
          return np.random.normal(0, 1, out_shape)

  def linear_classifier(self, features, scope = "lin_class", stddev=0.02, reuse=False):
      with tf.variable_scope(scope) as sc:
          w = tf.get_variable('w', [features.get_shape()[-1], 1],
                              initializer=tf.random_normal_initializer(stddev=stddev))
          b = tf.get_variable('biases', 1, initializer=tf.constant_initializer(0.0))
          logits = tf.matmul(features, w) + b
          return   tf.nn.sigmoid(logits), logits

  def z_mapping_function(self, z, output_channel, scope='z_mapping', act="relu", stddev=0.02, z_dim=None):
      with tf.variable_scope(scope) as sc:
          if z_dim is None:
            z_dim = z.get_shape()[-1]
          w = tf.get_variable('w', [z_dim, output_channel * 2],
                              initializer=tf.random_normal_initializer(stddev=stddev))
          b = tf.get_variable('biases', output_channel * 2, initializer=tf.constant_initializer(0.0))
          if act == "relu":
              out = tf.nn.relu(tf.matmul(z, w) + b)
          else:
              out = lrelu(tf.matmul(z, w) + b)
          return out[:, :output_channel], out[:, output_channel:]

#=======================================================================================================================
  def discriminator_IN(self, image,  cont_dim, reuse=False):
      if str(cfg["add_D_noise"]) == "true":
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
          h1 = lrelu(instance_norm(conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv'),'d_in1'))
          h2 = lrelu(instance_norm(conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv'),'d_in2'))
          h3 = lrelu(instance_norm(conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv'),'d_in3'))

          #Returning logits to determine whether the images are real or fake
          h4 = linear(slim.flatten(h3), 1, 'd_h4_lin')

          # Recognition network for latent variables has an additional layer
          encoder = lrelu((linear(slim.flatten(h3), 128, 'd_latent')))
          cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          return tf.nn.sigmoid(h4), h4, tf.nn.tanh(cont_vars)

  def discriminator_IN_style_res128(self, image, cont_dim, reuse=False):
      batch_size = tf.shape(image)[0]
      if str(cfg["add_D_noise"]) == "true":
          image = image + tf.random_normal(tf.shape(image), stddev=0.02)

      with tf.variable_scope("discriminator") as scope:
          if reuse:
              scope.reuse_variables()

          h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

          h1 = conv2d_specNorm(h0, self.df_dim * 2, name='d_h1_conv')
          h1, h1_mean, h1_var = instance_norm(h1, 'd_in1', True)
          # h1_mean = tf.reshape(h1_mean, (batch_size, self.df_dim * 2))
          # h1_var = tf.reshape(h1_var, (batch_size, self.df_dim * 2))
          # d_h1_style = tf.concat([h1_mean, h1_var], 0)
          # d_h1, d_h1_logits = self.linear_classifier(d_h1_style, "d_h1_class")
          h1 = lrelu(h1)

          h2 = conv2d_specNorm(h1, self.df_dim * 4, name='d_h2_conv')
          h2, h2_mean, h2_var = instance_norm(h2, 'd_in2', True)
          # h2_mean = tf.reshape(h2_mean, (batch_size, self.df_dim * 4))
          # h2_var = tf.reshape(h2_var, (batch_size, self.df_dim * 4))
          # d_h2_style = tf.concat([h2_mean, h2_var], 0)
          # d_h2, d_h2_logits = self.linear_classifier(d_h2_style, "d_h2_class")
          h2 = lrelu(h2)

          h3 = conv2d_specNorm(h2, self.df_dim * 8, name='d_h3_conv')
          h3, h3_mean, h3_var = instance_norm(h3, 'd_in3', True)
          # h3_mean = tf.reshape(h3_mean, (batch_size, self.df_dim * 8))
          # h3_var = tf.reshape(h3_var, (batch_size, self.df_dim * 8))
          # d_h3_style = tf.concat([h3_mean, h3_var], 0)
          # d_h3, d_h3_logits = self.linear_classifier(d_h3_style, "d_h3_class")
          h3 = lrelu(h3)

          h4 = conv2d_specNorm(h3, self.df_dim * 16, name='d_h4_conv')
          h4, h4_mean, h4_var = instance_norm(h4, 'd_in4', True)
          # h4_mean = tf.reshape(h4_mean, (batch_size, self.df_dim * 16))
          # h4_var = tf.reshape(h4_var, (batch_size, self.df_dim * 16))
          # d_h4_style = tf.concat([h4_mean, h4_var], 0)
          # d_h4, d_h4_logits = self.linear_classifier(d_h4_style, "d_h4_class")
          h4 = lrelu(h4)

          #Returning logits to determine whether the images are real or fake
          h5 = linear(slim.flatten(h4), 1, 'd_h5_lin')

          # Recognition network for latent variables has an additional layer
          # encoder = lrelu((linear(slim.flatten(h4), 128, 'd_latent')))
          # cont_vars = linear(encoder, cont_dim, "d_latent_prediction")

          # return tf.nn.sigmoid(h5), h5, tf.nn.tanh(cont_vars), d_h1_logits, d_h2_logits, d_h3_logits, d_h4_logits
          return tf.nn.sigmoid(h5), h5

  def generator_AdaIN(self, z, view_in, reuse=False):
      batch_size = tf.shape(z)[0]
      s_h, s_w, s_d = 64, 64, 64
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1))
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = AdaIn(w_tile, s0, b0)
              h0 = tf.nn.relu(h0)

          h1= deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_d=3, k_w=3, name='g_h1')
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1')
          h1 = AdaIn(h1, s1, b1)
          h1 = tf.nn.relu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_d=3, k_w=3, name='g_h2')
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2')
          h2 = AdaIn(h2, s2, b2)
          h2 = tf.nn.relu(h2)

          #=============================================================================================================
          h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)
          #=============================================================================================================
          # Collapsing depth dimension
          h2_2d = tf.reshape(h2_rotated, [batch_size, s_h4, s_w4, 16 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = tf.nn.relu(h3)
          #=============================================================================================================

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4], k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = AdaIn(h4, s4, b4)
          h4  = tf.nn.relu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = AdaIn(h5, s5, b5)
          h5 = tf.nn.relu(h5)

          h6 = deconv2d(h5, [batch_size, s_h, s_w, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h6')

          output = tf.nn.tanh(h6, name="output")
          # output = tf.nn.sigmoid(h6, name="output")
          return output

  def object_generator(self, z, view_in, obj_type, reuse=False):
      gf_dim = None
      variable_scope_name = None
      if obj_type == 'fg':
        variable_scope_name = 'fg_object_generator'
        w_dim = self.gf_dim # 64
      elif obj_type == 'bg':
        variable_scope_name = 'bg_object_generator'
        w_dim = self.gf_dim // 2 # 32

      batch_size, num_objs, z_dim = tf.shape(z)[0], tf.shape(z)[1], tf.shape(z)[2]
      s_h, s_w, s_d = 64, 64, 64 # height, width, depth of something
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope(variable_scope_name) as scope:
        if reuse:
          scope.reuse_variables()
        
        with tf.variable_scope('g_w_constant') as scope:
          w = tf.get_variable('w', [s_h16, s_w16, s_d16, w_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
          w_tile = tf.tile(tf.reshape(w, (1, 1, s_h16, s_w16, s_d16, w_dim * 8)), (batch_size, num_objs, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
          
          # reshape all objects of al samples as one big batch
          w_tile = tf.reshape(w_tile, (batch_size * num_objs, s_h16, s_w16, s_d16, w_dim * 8)) # (bsz, num_objects, h, w, d, c) -> (bsz * num_objects, h, w, d, c)
          z_dim = z.get_shape()[-1] # have to get before reshape, because reshape leaves it as None
          z = tf.reshape(z, (batch_size * num_objs, z_dim)) # (bsz, num_objects, z_dim) -> (bsz * num_objects, z_dim)

          s0, b0 = self.z_mapping_function(z, w_dim * 8, 'g_z0', z_dim=z_dim)
          h0 = AdaIn(w_tile, s0, b0)
          h0 = lrelu(h0)

        h1= deconv3d(h0, [batch_size * num_objs, s_h8, s_w8, s_d8, self.gf_dim * 2], k_h=3, k_w=3, k_d=3, name='g_h1')
        s1, b1 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z1', z_dim=z_dim)
        h1 = AdaIn(h1, s1, b1)
        h1 = lrelu(h1)

        h2 = deconv3d(h1, [batch_size * num_objs, s_h4, s_w4, s_d4, self.gf_dim * 1],  k_h=3, k_w=3, k_d=3, name='g_h2')
        s2, b2 = self.z_mapping_function(z, self.gf_dim * 1, 'g_z2', z_dim=z_dim)
        h2 = AdaIn(h2, s2, b2)
        h2 = lrelu(h2)

        h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
        h2_rotated = transform_voxel_to_match_image(h2_rotated)
        h2_out = h2_rotated

        h2_proj1 = deconv3d(h2_rotated, [batch_size * num_objs, s_h4, s_w4, s_d4, self.gf_dim], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')
        h2_proj1 = lrelu( h2_proj1)

        h2_proj2 = deconv3d(h2_proj1, [batch_size * num_objs, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')
        h2_proj2 = lrelu( h2_proj2)
        h2_out = h2_proj2

        # expand objects
        # print(h2_out.shape)
        return tf.reshape(h2_out, (batch_size, num_objs, *h2_out.shape[1:]))


  def generator_AdaIN_block(self, z_bg, z_fg, view_in_bg, view_in_fg, reuse=False):
      bsz, num_objs_bg, z_dim_bg = tf.shape(z_bg)[0], tf.shape(z_bg)[1], tf.shape(z_bg)[2]
      _, num_objs_fg, z_dim_fg = tf.shape(z_fg)[0], tf.shape(z_fg)[1], tf.shape(z_fg)[2]

      s_h, s_w, s_d = 64, 64, 64
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()

          h2_rotated_bg = self.object_generator(z_bg, view_in_bg, 'bg', reuse)
          h2_rotated_fg = self.object_generator(z_fg, view_in_fg, 'fg', reuse)
          h2_rotated = tf.concat((h2_rotated_bg, h2_rotated_fg), axis=1, name='h2_rotated')
          h2_rotated_agg = tf.math.reduce_max(h2_rotated, axis=1, name='h2_rotated_agg')
          h2_2d = tf.reshape(h2_rotated_agg, [bsz, s_h4, s_w4, s_d4 * self.gf_dim])

          pop1 = tf.print('h2_2d', h2_2d, output_stream=sys.stdout)

          # 1X1 convolution
          h3 = deconv2d(h2_2d, [bsz, s_h4, s_w4, self.gf_dim], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = lrelu(h3)

          pop2 = tf.print('h3', h3, output_stream=sys.stdout)

          h4  = deconv2d(h3, [bsz, s_h2, s_w2, self.gf_dim],  k_h=4, k_w=4, name='g_h4')
          # s4, b4 = self.z_mapping_function(z, self.gf_dim, 'g_z4')
          # h4  = AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          pop3 = tf.print('h4', h4, output_stream=sys.stdout)

          h5 = deconv2d(h4, [bsz, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          # s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          # h5 = AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          pop4 = tf.print('h5', h5, output_stream=sys.stdout)
          
          h6 = deconv2d(h5, [bsz, s_h * 2, s_w * 2, self.gf_dim], k_h=4, k_w=4, name='g_h6')
          # s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          # h6 = AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          pop5 = tf.print('h6', h6, output_stream=sys.stdout)

          h7 = deconv2d(h6, [bsz, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          pop6 = tf.print('h7', h7, output_stream=sys.stdout)

          with tf.control_dependencies([pop1, pop2, pop3, pop4, pop5, pop6]):
              output = tf.nn.tanh(h7, name="output")
              return output


  def generator_AdaIN_res128(self, z_bg, z_fg, view_in_bg, view_in_fg, reuse=False):
      batch_size, num_objs_bg, z_dim_bg = tf.shape(z_bg)
      _, num_objs_fg, z_dim_fg = tf.shape(z_fg)

      s_h, s_w, s_d = 64, 64, 64
      s_h2, s_w2, s_d2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2), conv_out_size_same(s_d, 2)
      s_h4, s_w4, s_d4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2), conv_out_size_same(s_d2, 2)
      s_h8, s_w8, s_d8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2), conv_out_size_same(s_d4, 2)
      s_h16, s_w16, s_d16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2), conv_out_size_same(s_d8, 2)

      with tf.variable_scope("generator") as scope:
          if reuse:
              scope.reuse_variables()
          #A learnt constant "template"
          with tf.variable_scope('g_w_constant'):
              w = tf.get_variable('w', [s_h16, s_w16, s_d16, self.gf_dim * 8], initializer=tf.random_normal_initializer(stddev=0.02))
              w_tile = tf.tile(tf.expand_dims(w, 0), (batch_size, 1, 1, 1, 1)) #Repeat the learnt constant features to make a batch
              s0, b0 = self.z_mapping_function(z, self.gf_dim * 8, 'g_z0')
              h0 = AdaIn(w_tile, s0, b0)
              h0 = lrelu(h0)

          h1 = deconv3d(h0, [batch_size, s_h8, s_w8, s_d8, self.gf_dim * 4], k_h=3, k_w=3, k_d=3, name='g_h1')
          s1, b1 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z1')
          h1 = AdaIn(h1, s1, b1)
          h1 = lrelu(h1)

          h2 = deconv3d(h1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 2],  k_h=3, k_w=3, k_d=3, name='g_h2')
          s2, b2 = self.z_mapping_function(z, self.gf_dim * 2, 'g_z2')
          h2 = AdaIn(h2, s2, b2)
          h2 = lrelu(h2)

          #=============================================================================================================
          h2_rotated = tf_3D_transform(h2, view_in, 16, 16)
          h2_rotated = transform_voxel_to_match_image(h2_rotated)

          h2_proj1 = deconv3d(h2_rotated, [batch_size, s_h4, s_w4, s_d4, self.gf_dim * 1], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='g_h2_proj1')
          h2_proj1 = lrelu( h2_proj1)

          h2_proj2 = deconv3d(h2_proj1, [batch_size, s_h4, s_w4, s_d4, self.gf_dim ], k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1,  name='g_h2_proj2')
          h2_proj2 = lrelu( h2_proj2)
          # =============================================================================================================
          # Collapsing depth dimension
          h2_2d = tf.reshape(h2_proj2, [batch_size, s_h4, s_w4, s_d4 * self.gf_dim])
          # 1X1 convolution
          h3 = deconv2d(h2_2d, [batch_size, s_h4, s_w4, self.gf_dim * 16 // 2], k_h=1, k_w=1, d_h=1, d_w=1, name='g_h3')
          h3 = lrelu(h3)
          # =============================================================================================================

          h4  = deconv2d(h3, [batch_size, s_h2, s_w2, self.gf_dim * 4],  k_h=4, k_w=4, name='g_h4')
          s4, b4 = self.z_mapping_function(z, self.gf_dim * 4, 'g_z4')
          h4  = AdaIn(h4, s4, b4)
          h4 = lrelu(h4)

          h5 = deconv2d(h4, [batch_size, s_h, s_w, self.gf_dim], k_h=4, k_w=4, name='g_h5')
          s5, b5 = self.z_mapping_function(z, self.gf_dim, 'g_z5')
          h5 = AdaIn(h5, s5, b5)
          h5 = lrelu(h5)

          h6 = deconv2d(h5, [batch_size, s_h * 2, s_w * 2, self.gf_dim // 2], k_h=4, k_w=4, name='g_h6')
          s6, b6 = self.z_mapping_function(z, self.gf_dim // 2, 'g_z6')
          h6 = AdaIn(h6, s6, b6)
          h6 = lrelu(h6)

          h7 = deconv2d(h6, [batch_size, s_h * 2, s_w * 2, self.c_dim], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h7')

          output = tf.nn.tanh(h7, name="output")
          # output = tf.nn.sigmoid(h7, name="output")
          return output

#=======================================================================================================================
  @property
  def model_dir(self):
    return "{}_{}_{}".format(
        self.dataset_name,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "HoloGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0


