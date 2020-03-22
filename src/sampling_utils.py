import math
import torch


def sample_z(batch_size, z_dim, num_objects=(3, 10)):
    if type(num_objects) is tuple:
        num_objs = torch.randint(num_objects[0], num_objects[1] + 1, (batch_size,),)
        tensors = []
        max_len = max(num_objs)
        for no in num_objs:
            _t = uniform((no, z_dim))
            _z = torch.zeros((max_len - no, z_dim))
            _t = torch.cat((_t, _z), dim=0)
            tensors.append(_t)
        z = torch.stack(tensors, axis=0)
        return z
    else:
        return uniform((batch_size, num_objects, z_dim), -1, 1)


def uniform(shape, low=-1, high=1):
    return torch.empty(shape).uniform_(low, high)


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
    azimuth = uniform(
        (batch_size, num_objects, 1),
        math.radians(azimuth_range[0]),
        math.radians(azimuth_range[1]),
    )
    elevation = uniform(
        (batch_size, num_objects, 1),
        math.radians(elevation_range[0]),
        math.radians(elevation_range[1]),
    )
    roll = uniform(
        (batch_size, num_objects, 1),
        math.radians(roll_range[0]),
        math.radians(roll_range[1]),
    )
    scale = uniform((batch_size, num_objects, 1), scale_range[0], scale_range[1])
    # scale_y = uniform((batch_size, num_objects, 1), scale_range[0], scale_range[1])
    # scale_z = uniform((batch_size, num_objects, 1), scale_range[0], scale_range[1])
    tx = uniform((batch_size, num_objects, 1), tx_range[0], tx_range[1])
    ty = uniform((batch_size, num_objects, 1), ty_range[0], ty_range[1])
    tz = uniform((batch_size, num_objects, 1), tz_range[0], tz_range[1])

    return torch.cat(
        [azimuth, elevation, roll, scale, scale, scale, tx, ty, tz], dim=2,
    )


def sample_z_and_view(batch_size, z_dim, num_objects):
    z = sample_z(batch_size, z_dim, num_objects)
    view = sample_view(batch_size, z.shape[1])

    return z, view
