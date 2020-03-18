from __future__ import division
from __future__ import print_function

import os

# import os.path as osp

import yaml
import json
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C
__C.experiment_name = ""
__C.run_name = ""
__C.comet_project_name = ""
__C.logcomet = False

__C.train = edict(
    batch_size=64,
    epochs=50,
    snapshot_interval=5,
    generator=edict(lr=0.0001, beta1=0.5, beta2=0.999, update_freq=2,),
    discriminator=edict(lr=0.0001, beta1=0.5, beta2=0.999, random_noise=False,),
)

__C.model = edict(
    generator=edict(
        z_dim_bg=30,
        z_dim_fg=90,
        w_dim_bg=256,
        w_dim_fg=512,
        filters=[64, 64, 64],
        ks=[1, 4, 4],
        strides=[1, 2, 2],
    ),
    discriminator=edict(
        filters=[64, 128, 256, 512], ks=[5, 5, 5, 5], strides=[2, 2, 2, 2],
    ),
)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            # raise KeyError('{} is not a valid config key'.format(k))
            print("{} is not a valid config key".format(k))
            continue

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif b[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(
                    ("Type mismatch ({} vs. {}) " "for config key: {}").format(
                        type(b[k]), type(v), k
                    )
                )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, "r") as f:
        _, ext = os.path.splitext(filename)
        if ext == ".yml" or ext == ".yaml":
            file_cfg = edict(yaml.safe_load(f))
        elif ext == ".json":
            file_cfg = edict(json.load(f))

    _merge_a_into_b(file_cfg, __C)
