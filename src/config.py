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
__C.cuda = False
__C.gpu_id = '-1'
__C.num_workers = 0

__C.train = edict(
    data_dir="",
    resume='',
    batch_size=4,
    epochs=20,
    snapshot_interval=5,
    it_log_interval=2000,
    generator=edict(
        optimizer=edict(lr=0.0001, betas=(0.5, 0.999)),
    ),
    scene_encoder=edict(
        optimizer=edict(lr=0.0001, betas=(0.5, 0.999)),
        random_noise=False,
        scale_prior=(-2, 0.1),
        center_prior=(0, 0.1),
        presence_prior=3e-2,
        kl_weight=1e-6,
    ),
)

__C.model = edict(
    generator=edict(
        z_dim_bg=30,
        z_dim_fg=90,
        w_dim_bg=256,
        w_dim_fg=512,
        filters=[64, 64],
        ks=[4, 4],
        strides=[2, 2],
        use_learnable_proj=False,
        use_inverse_transform=False,
    ),
    scene_encoder=edict(
        ncs=[16, 32, 32, 64, 128, 256],
        kss=[3, 4, 3, 4, 4, 4, 4],
        ss=[1, 2, 1, 2, 2, 1],
        ngs=[4, 8, 4, 8, 8, 16],
        out_proj_dim=64,
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
