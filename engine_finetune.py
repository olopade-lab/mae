# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup

import util.lr_sched as lr_sched
import util.misc as misc


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    # This was supposed to help with uneven inputs, but I still can't get it working
    # https://github.com/pytorch/pytorch/pull/42577/files#diff-cf805caec69e0dd5e7ef09540316fdd9eefe407f317c4d89ebb1c2f82167045b
    # with model.join():
    for data_iter_step, (x, contralateral_x, duration, event) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # x = x.to(device, non_blocking=True)
        # contralateral_x = contralateral_x.to(device, non_blocking=True)
        duration = duration.to(device, non_blocking=True)
        event = event.to(device, non_blocking=True)

        # TODO implement
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(x, contralateral_x, device)
            loss = criterion(outputs, duration, event)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, c_index, auc, tag="val"):

    metric_logger = misc.MetricLogger(delimiter="  ")
    print("starting evaluate")

    # switch to evaluation mode
    model.eval()
    print("switching to eval mode")

    for batch in metric_logger.log_every(data_loader, 10, tag):
        x, contralateral_x, duration, years_to_event, event, study_id = batch

        # x = x.to(device, non_blocking=True)
        # contralateral_x = contralateral_x.to(device, non_blocking=True)
        duration = duration.to(device, non_blocking=True)
        event = event.to(device, non_blocking=True)
        years_to_event = years_to_event.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(x, contralateral_x, device)
            loss = criterion(output, duration, event)

        c_index.update(output, duration, event)
        auc.update(output, duration, event, years_to_event)
        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    c_index_value = c_index.compute()
    # auc_results = auc.compute()
    c_index.reset()
    # auc.reset()

    print(
        "{tag}: c-index {c_index:.3f} loss {losses.global_avg:.3f}".format(
            tag=tag, c_index=c_index_value, losses=metric_logger.loss
        )
    )

    # TODO fix to deduplicate
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["c_index"] = c_index_value
    # for k, v in auc_results.items():
    #     stats[k] = v

    return stats
