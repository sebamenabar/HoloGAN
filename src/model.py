import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    LeakyReLU,
    Conv2DTranspose,
    Conv3DTranspose,
)

from layers import AdaIN, InstanceNorm, SpectralNorm
from transformation_utils import tf_3d_transform, transform_voxel_to_match_image


class ObjectGenerator(Model):
    def __init__(
        self,
        z_dim,
        w_dim,
        use_learnable_proj=True,  # TEMP: to replace with perspective projection
        w_shape=(4, 4, 4),
        upconv_flters=[128, 64],
        upconv_ks=[3, 3],  # for upconvolution, ks: kernel_size
        upconv_strides=[2, 2],
    ):
        super().__init__()

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.w_shape = w_shape
        self.use_learnable_proj = use_learnable_proj

        self.adain0 = AdaIN(w_dim, z_dim)
        self.lrelu = LeakyReLU(alpha=0.2)
        self.w = tf.Variable(
            tf.random.normal((*w_shape, w_dim), stddev=0.02), trainable=True
        )

        self.deconvs = []
        self.adains = []
        for filters, ks, stride in zip(upconv_flters, upconv_ks, upconv_strides):
            self.deconvs.append(
                Conv3DTranspose(
                    filters=filters,
                    kernel_size=ks,
                    strides=stride,
                    padding="same",
                    kernel_initializer=tf.initializers.RandomNormal(stddev=0.02),
                    bias_initializer="zeros",
                )
            )
            self.adains.append(AdaIN(filters, z_dim))

        if use_learnable_proj:
            self.proj1 = K.layers.Conv3DTranspose(
                filters=upconv_flters[-1],
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=tf.initializers.RandomNormal(stddev=0.02),
                bias_initializer="zeros",
            )
            self.proj2 = K.layers.Conv3DTranspose(
                filters=upconv_flters[-1],
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
                kernel_initializer=tf.initializers.RandomNormal(stddev=0.02),
                bias_initializer="zeros",
            )
        else:
            print('Using 3D affine transformations for objects')

    def call(self, z, view_in=None):
        """
        z: (bsz, num_objs, z_dim)
        view_in: (bsz, num_objs, 6)
        """
        z_dim = z.shape[-1]
        assert (
            z_dim == self.z_dim
        ), f"Input dimension ({z_dim}) does not match expected value ({self.z_dim})"

        if len(z.shape) == 3:
            bsz, num_objs, _ = z.shape
            z = tf.reshape(z, (bsz * num_objs, self.z_dim))
        else:
            bsz, _ = z.shape
            num_objs = 1
        if view_in is not None and len(view_in.shape) == 3:
            view_in = tf.reshape(view_in, (bsz * num_objs, 9))

        w = tf.repeat(tf.expand_dims(self.w, 0), bsz * num_objs, axis=0, name="w")
        h = self.adain0(w, z)
        h = self.lrelu(h)

        for deconv, adain in zip(self.deconvs, self.adains):
            h = deconv(h)
            h = adain(h, z)
            h = self.lrelu(h)

        # TODO 3d perspective transform
        # TEMP replace perspective projection with learnable projection
        if self.use_learnable_proj:
            h2_proj1 = self.proj1(h)
            h2_proj1 = self.lrelu(h2_proj1)

            h2_proj2 = self.proj2(h2_proj1)
            h2_proj2 = self.lrelu(h2_proj2)

            out = h2_proj2
        else:
            h_rotated = tf_3d_transform(h, view_in, 16, 16)
            h_rotated = transform_voxel_to_match_image(h_rotated)

            out = h_rotated

        return tf.reshape(out, (bsz, num_objs, *out.shape[1:]))


class Generator(Model):
    def __init__(
        self,
        z_dim_bg=30,
        z_dim_fg=90,
        w_dim_bg=256,
        w_dim_fg=512,
        filters=[64, 64, 64],
        ks=[1, 4, 4],
        strides=[1, 2, 2],
        use_learnable_proj=True
    ):
        super().__init__()

        self.z_dim_bg = z_dim_bg
        self.z_dim_fg = z_dim_fg
        # self.lrelu = K.layers.LeakyReLU(alpha=0.2)

        self.lrelu = LeakyReLU(alpha=0.2)
        self.bg_generator = ObjectGenerator(z_dim_bg, w_dim_bg, use_learnable_proj)
        self.fg_generator = ObjectGenerator(z_dim_fg, w_dim_fg, use_learnable_proj)

        self.deconvs = []
        for f, k, s in zip(filters, ks, strides):
            self.deconvs.append(
                Conv2DTranspose(
                    filters=f,
                    kernel_size=k,
                    strides=s,
                    padding="same",
                    kernel_initializer=tf.initializers.RandomNormal(stddev=0.02),
                    bias_initializer="zeros",
                )
            )

        self.out_conv = Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=1,
            padding="same",
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.02),
            bias_initializer="zeros",
        )

    @tf.function
    def call(self, z_bg, z_fg, view_in_bg, view_in_fg):
        bsz = z_bg.shape[0]

        bg = self.bg_generator(
            z_bg, view_in_bg,
        )  # (bsz, 1, height, width, depth, num_channels)
        fg = self.fg_generator(
            z_fg, view_in_fg,
        )  # (bsz, num_objects, height, width, depth, num_channels)
        composed_scene = tf.concat((bg, fg), axis=1)
        composed_scene = tf.math.reduce_max(
            composed_scene, axis=1, name="composed_scene"
        )

        h2_2d = tf.reshape(composed_scene, (bsz, 16, 16, 16 * 64))
        h = h2_2d
        for deconv in self.deconvs:
            h = self.lrelu(deconv(h))

        h = tf.math.tanh(self.out_conv(h))

        return h


class Discriminator(Model):
    def __init__(
        self, filters=[64, 128, 256, 512], ks=[5, 5, 5, 5], strides=[2, 2, 2, 2],
    ):
        super().__init__()

        self.flatten = Flatten()
        self.lrelu = LeakyReLU(alpha=0.2)
        self.convs = []
        self.inst_norms = []
        for f, k, s in zip(filters, ks, strides):
            self.convs.append(
                Conv2D(
                    filters=f,
                    kernel_size=k,
                    strides=s,
                    padding="SAME",
                    kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                    bias_initializer="zeros",
                    kernel_constraint=SpectralNorm(),
                )
            )
            self.inst_norms.append(InstanceNorm(f))
        self.linear = Dense(1, kernel_constraint=SpectralNorm())

    @tf.function
    def call(self, x):
        for conv, norm in zip(self.convs, self.inst_norms):
            x = self.lrelu(norm(conv(x)))

        return self.linear(self.flatten(x))
