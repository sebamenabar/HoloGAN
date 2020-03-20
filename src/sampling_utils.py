import math
import tensorflow as tf


def sample_z(batch_size, z_dim, num_objects=(3, 10)):
    if type(num_objects) is tuple:
        num_objs = tf.random.uniform(
            (batch_size,),
            minval=num_objects[0],
            maxval=num_objects[1] + 1,
            dtype=tf.int32,
        )
        tensors = []
        max_len = max(num_objs)
        for no in num_objs:
            _t = tf.random.uniform((no, z_dim), minval=-1, maxval=1)
            _z = tf.zeros((max_len - no, z_dim), dtype=tf.float32)
            _t = tf.concat((_t, _z), axis=0)
            tensors.append(_t)
        z = tf.stack(tensors, axis=0)
        return z
    else:
        return tf.random.uniform((batch_size, num_objects, z_dim), minval=-1, maxval=1)


def sample_view(
    batch_size,
    num_objects,
    azimuth_range=(-180, 180),
    elevation_range=(0, 0),
    roll_range=(0.0, 0.0),
    scale_range=(0.5, 1.0),
    tx_range=(-5, 5),
    ty_range=(0, 0),
    tz_range=(-5, 5),
):
    azimuth = tf.random.uniform(
        (batch_size, num_objects, 1),
        minval=math.radians(azimuth_range[0]),
        maxval=math.radians(azimuth_range[1]),
    )
    elevation = tf.random.uniform(
        (batch_size, num_objects, 1),
        minval=math.radians(elevation_range[0]),
        maxval=math.radians(elevation_range[1]),
    )
    roll = tf.random.uniform(
        (batch_size, num_objects, 1),
        minval=math.radians(roll_range[0]),
        maxval=math.radians(roll_range[1]),
    )
    scale = tf.random.uniform(
        (batch_size, num_objects, 1), minval=scale_range[0], maxval=scale_range[1]
    )
    # scale_y = tf.random.uniform((batch_size, 1), minval=azimuth_range[0], maxval=azimuth_range[1])
    # scale_z = tf.random.uniform((batch_size, 1), minval=azimuth_range[0], maxval=azimuth_range[1])
    tx = tf.random.uniform(
        (batch_size, num_objects, 1), minval=tx_range[0], maxval=tx_range[1]
    )
    ty = tf.random.uniform(
        (batch_size, num_objects, 1), minval=ty_range[0], maxval=ty_range[1]
    )
    tz = tf.random.uniform(
        (batch_size, num_objects, 1), minval=tz_range[0], maxval=tz_range[1]
    )

    return tf.concat(
        [azimuth, elevation, roll, scale, scale, scale, tx, ty, tz],
        axis=2,
    )


def sample_z_and_view(batch_size, z_dim, num_objects):
    z = sample_z(batch_size, z_dim, num_objects)
    view = sample_view(batch_size, z.shape[1])

    return z, view
