import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import Constraint

POWER_ITERATIONS = 1


class AdaIN(Model):
    def __init__(self, feat_dim, z_dim):
        super().__init__()
        self.z_proj = Dense(
            units=feat_dim * 2,
            input_shape=(z_dim,),
            activation="8relu",
            use_bias=True,
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.02),
            bias_initializer="zeros",
        )

    def call(self, features, z):
        """
        Adaptive instance normalization component. Works with both 4D and 5D tensors
        :features: features to be normalized
        :scale: scaling factor. This would otherwise be calculated as the sigma from a "style" features in style transfer
        :bias: bias factor. This would otherwise be calculated as the mean from a "style" features in style transfer
        """
        scale, bias = tf.split(self.z_proj(z), 2, num=2, axis=1)
        mean, variance = tf.nn.moments(
            features, list(range(len(features.get_shape())))[1:-1], keepdims=True
        )  # Only consider spatial dimension
        sigma = tf.math.rsqrt(variance + 1e-8)
        normalized = (features - mean) * sigma
        scale_broadcast = tf.reshape(scale, tf.shape(mean))
        bias_broadcast = tf.reshape(bias, tf.shape(mean))
        normalized = scale_broadcast * normalized
        normalized += bias_broadcast
        return normalized


def l2_normalize(x, eps=1e-12):
    """
  Scale input by the inverse of it's euclidean norm
  """
    return x / tf.linalg.norm(x + eps)


class SpectralNorm(Constraint):
    """
    Uses power iteration method to calculate a fast approximation
    of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm
    """

    def __init__(self, power_iters=POWER_ITERATIONS):
        self.n_iters = power_iters

    def __call__(self, w):
        flattened_w = tf.reshape(w, [w.shape[0], -1])
        u = tf.random.normal([flattened_w.shape[0]])
        v = tf.random.normal([flattened_w.shape[1]])
        for i in range(self.n_iters):
            v = tf.linalg.matvec(tf.transpose(flattened_w), u)
            v = l2_normalize(v)
            u = tf.linalg.matvec(flattened_w, v)
            u = l2_normalize(u)
        sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
        return w / sigma

    def get_config(self):
        return {"n_iters": self.n_iters}


class InstanceNorm(Model):
    def __init__(self, num_channels):
        super().__init__()
        self.scale = tf.Variable(
            initial_value=tf.random.normal((num_channels,), 1.0, 0.02)
        )
        self.offset = tf.Variable(
            initial_value=tf.zeros((num_channels,), dtype=tf.float32)
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset
