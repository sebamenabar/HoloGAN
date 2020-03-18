from __future__ import print_function

import os
import os.path as osp
import sys
import shutil
import random
import dateutil
import argparse
from datetime import datetime as dt

import tensorflow as tf

import comet_ml
from dotenv import load_dotenv

from misc_utils import mkdir_p
from config import cfg, cfg_from_file


def parse_args():
    parser = argparse.ArgumentParser()

    # Config
    parser.add_argument(
        "--cfg", dest="cfg_file", help="optional config file", default=None, type=str
    )
    parser.add_argument("--manual-seed", type=int, help="manual seed")

    # Resume training
    parser.add_argument('--bsz', type=float)
    parser.add_argument("--resume", type=str)

    # Logs
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--comet-project-name", type=str)
    parser.add_argument("--logcomet", action="store_true")
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--run-name", type=str)
    parser.add_argument('--no-log', action='store_true')

    args = parser.parse_args()
    return args


def set_logdir(args, experiment_name, run_name=""):
    now = dt.now(dateutil.tz.tzlocal())
    now = now.strftime("%m-%d-%Y-%H-%M-%S")
    if run_name:
        run_name = run_name + "-" + now
    else:
        run_name = now
    logdir = osp.join("data", experiment_name, run_name)
    mkdir_p(logdir)
    print("Saving output to: {}".format(logdir))
    code_dir = os.path.join(os.getcwd(), "src")
    mkdir_p(os.path.join(logdir, "src"))
    for filename in os.listdir(code_dir):
        if filename.endswith(".py"):
            shutil.copy(code_dir + "/" + filename, os.path.join(logdir, "src"))
    shutil.copy(args.cfg_file, logdir)
    return logdir


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg.cfg_file = args.cfg_file
        cfg_from_file(args.cfg_file)
    if args.bsz:
        cfg.train.batch_size = args.bsz
    if args.experiment_name:
        cfg.experiment_name = args.experiment_name
    if args.run_name:
        cfg.run_name = args.run_name

    manual_seed = (
        args.manual_seed or getattr(cfg, "manual_seed") or random.randint(1, 10000)
    )
    random.seed(manual_seed)
    tf.random.set_seed(manual_seed)
    cfg.manual_seed = manual_seed
    
    log = not args.no_log
    cfg.no_log = args.no_log
    cfg.logcomet = args.logcomet & log
    if log and cfg.experiment_name:
        logdir = set_logdir(cfg, cfg.experiment_name, cfg.run_name)
    else:
        logdir = None
    