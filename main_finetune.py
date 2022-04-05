# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import datetime
import json
import os

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import random
import string
import time
from pathlib import Path
from tkinter import N

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from maicara.models.metrics import ConcordanceIndex
from maicara.preprocessing.utils import log_code_state
from pycox.models import logistic_hazard
from torch.utils.tensorboard import SummaryWriter

assert timm.__version__ == "0.3.2"  # version check
from maicara.data.chimec import load_datasets
from maicara.data.constants import CHIMEC_MEAN, CHIMEC_STD
from maicara.models.utils import (
    BalancedClassSampler,
    DistributedSamplerWrapper,
    wandb_bool,
)
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.layers import trunc_normal_

import models_vit
import util.lr_decay as lrd
import util.misc as misc
import wandb
from engine_finetune import evaluate, train_one_epoch
from util.datasets import build_transform
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--img_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    # Dataset parameters
    parser.add_argument(
        "--output_length",
        default=1000,
        type=int,
        help="Length of the output representation",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="path where to save, empty with wandb enabled will save to wandb dir",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--test_size", default=0.2, type=float)
    parser.add_argument("--val_size", default=0.2, type=float)
    parser.add_argument("--prescale", default=1, type=float)
    parser.add_argument("--balance", default=False, type=wandb_bool)
    parser.add_argument("--debug", default=False, type=wandb_bool)
    parser.add_argument(
        "--metadata_path",
        default="/gpfs/data/huo-lab/Image/annawoodard/maicara/data/interim/mammo_v10/clean_metadata.pkl",
    )

    parser.add_argument(
        "--project",
        default=None,
        help="wandb project to log to (None to disable wandb logging)",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="wandb group to log to",
    )
    parser.add_argument("--config", default=None, help="wandb config to load")
    parser.add_argument(
        "--gpus",
        default=None,
        type=str,
        help="Comma-separated list of GPUs to run on. `None` to run on all. Example: `--gpus 0,1,2`",
    )
    parser.add_argument(
        "--trials",
        default=1,
        type=int,
        help="Number of times to run",
    )

    return parser


def main(args):
    args.output_dir = misc.prepare_output_dir(args.output_dir, "finetuning", args.group)
    misc.setup_for_distributed(True, os.path.join(args.output_dir, "finetune.log"))

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device("cuda:0")

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    train_transform = build_transform(
        is_train=True, args=args, mean=(CHIMEC_MEAN,), std=(CHIMEC_STD,)
    )
    val_transform = build_transform(
        is_train=False, args=args, mean=(CHIMEC_MEAN,), std=(CHIMEC_STD,)
    )

    label_transform, train_dataset, val_dataset, test_dataset = load_datasets(
        args.metadata_path,
        args.prescale,
        args.debug,
        args.test_size,
        args.val_size,
        train_transform,
        val_transform,
    )
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    sampler_train = BalancedClassSampler(train_dataset.metadata.event.to_list())
    print("Sampler_train = %s" % str(sampler_train))
    sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    sampler_test = torch.utils.data.SequentialSampler(test_dataset)

    if global_rank == 0 and args.output_dir is not None and not args.eval:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # TODO implement for surv
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.output_length,
        )

    model = models_vit.__dict__[args.model](
        output_length=args.output_length,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        img_size=args.img_size,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {
                "head.weight",
                "head.bias",
                "fc_norm.weight",
                "fc_norm.bias",
            }
        else:
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.gpus is None:
        args.gpus = list(range(torch.cuda.device_count()))
    elif isinstance(args.gpus, str):
        args.gpus = [int(x) for x in args.gpus.split(",")]
    print(
        f"Initializing DataParallel to run on GPUs: {','.join([str(x) for x in args.gpus])}"
    )
    if len(args.gpus) > 1:
        model = torch.nn.parallel.DataParallel(model, device_ids=args.gpus)
    # I'm having issues with DDP, DP isn't SO much of a performance hit, see: https://spell.ml/blog/pytorch-distributed-data-parallel-XvEaABIAAB8Ars0e
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model,
        args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = logistic_hazard.NLLLogistiHazardLoss()
    # if mixup_fn is not None:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing > 0.0:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    test_c_index = ConcordanceIndex(cuts=label_transform.cuts).to(device)
    val_c_index = ConcordanceIndex(cuts=label_transform.cuts).to(device)

    if args.eval:
        test_stats = evaluate(
            data_loader_test,
            model,
            device,
            criterion,
            test_c_index,
            "test",
        )
        print(
            f"Concordance index of the network on the {len(test_dataset)} test exams: {test_c_index.compute():.2f}"
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args,
        )
        val_stats = evaluate(
            data_loader_val,
            model,
            device,
            criterion,
            val_c_index,
            "val",
        )
        c_index = val_c_index.compute()
        val_c_index.reset()

        if log_writer is not None:
            log_writer.add_scalar("val/c_index", val_stats["c_index"], epoch)
            log_writer.add_scalar("val/loss", val_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if log_writer is not None:
            log_writer.flush()
        with open(
            os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

    if args.output_dir:
        misc.save_model(
            args=args,
            model=model,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
        )

    if args.val_size > 0:
        evaluate(
            data_loader_test,
            model,
            device,
            criterion,
            test_c_index,
            "test",
        )
        c_index = test_c_index.compute()
    print(
        f"Concordance index of the network on the {len(test_dataset)} test exams: {c_index:.2f}"
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    return c_index


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.group = "".join([random.choice(string.ascii_lowercase) for _ in range(5)])
    metric_logger = misc.MetricLogger(delimiter="  ")
    for i in range(args.trials):
        print(f"starting trial {i}")
        args.seed = i
        metric_logger.update(c_index=main(args))
    print(metric_logger)
    if args.project is not None:
        wandb.login()
        config = args.config if args.config else args
        run = wandb.init(config=config, project=args.project, group=args.group)
        log_code_state(run.dir)
        c_index = metric_logger.c_index.global_avg
        std = metric_logger.c_index.std
        wandb.log(
            {
                "collated/c-index": c_index,
                "collated/weighted-c-index": c_index / std,
                "collated/c-index-std": std,
            }
        )
