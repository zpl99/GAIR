# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch
import wandb
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from util import eval_accuracy_fn
import csv

def train_one_epoch_multi_model_eco_data(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, log_writer=None,
                    args=None):

    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0: # TODO: WHAT IF I DO NOT USE ADJUST LEARNING RATE
           lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        # samples = samples.to(device, non_blocking=True)

        # input_data = samples["image"].to(device, non_blocking=True)
        rs_image = samples['rs_image'].to(device,non_blocking=True)
        sv_image = samples['sv_image'].to(device,non_blocking=True)
        sv_location = samples['sv_location'].to(device,non_blocking=True)
        bbox_information = samples['bbox_information'].to(device,non_blocking=True)
        target_data = samples["values"].to(device, non_blocking=True)



        with torch.cuda.amp.autocast():
            outputs = model(rs_image,sv_image,sv_location,bbox_information)
            loss = criterion(outputs, target_data)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_eco_data(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, log_writer=None,
                    args=None):

    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0: # TODO: WHAT IF I DO NOT USE ADJUST LEARNING RATE
           lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        # samples = samples.to(device, non_blocking=True)

        input_data = samples["image"].to(device, non_blocking=True)
        target_data = samples["values"].to(device, non_blocking=True)


        with torch.cuda.amp.autocast():
            if args.benchmark == "imagenet" or args.benchmark =="street_view_imagenet":
                outputs = model(input_data)
                outputs = outputs["pooler_output"][:,0,:]
            elif args.benchmark == "croma":
                outputs = model(input_data)
                outputs = outputs[0]
            else:
                outputs = model(input_data)
            loss = criterion(outputs, target_data)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_meta_data(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, log_writer=None,
                    args=None):

    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        # samples = samples.to(device, non_blocking=True)

        input_data = samples["image"].to(device, non_blocking=True)
        target_data = samples["values"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            if args.benchmark == "imagenet" or args.benchmark =="street_view_imagenet":
                outputs = model(input_data)
                outputs = outputs["pooler_output"][:,0,:]
            else:
                outputs = model(input_data)
            loss = criterion(outputs, target_data)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_fmow(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):

    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        # samples = samples.to(device, non_blocking=True)

        input_data = samples["image"].to(device, non_blocking=True)
        target_data = samples["values"].to(device, non_blocking=True)
        #TODO: should I implement mixup?
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(input_data, target_data)

        with torch.cuda.amp.autocast():
            if args.benchmark == "imagenet" or args.benchmark =="street_view_imagenet":
                outputs = model(input_data)
                outputs = outputs["pooler_output"][:,0,:]
            elif args.benchmark == "croma":
                outputs,_ = model(input_data)
            else:
                outputs = model(input_data)
            loss = criterion(outputs, target_data)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_temporal(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, timestamps, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, timestamps)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': max_lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_meta_data(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    predictions = []
    true_values = []
    num_classes = None
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["image"]
        target = batch["values"]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output

        with torch.cuda.amp.autocast():
            if args.benchmark == "imagenet" or args.benchmark == "street_view_imagenet":
                output = model(images)
                output = output["pooler_output"][:,0,:]
            elif args.benchmark == "croma":
                output,_ = model(images)
            else:
                output = model(images)
            loss = criterion(output, target)

        predictions.append(output.detach().cpu())
        true_values.append(target.detach().cpu())

        num_classes = output.shape[-1]
        acc = eval_accuracy_fn.evaluate_small_classes_classification(output, target, num_classes)
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        if num_classes == 2:
            metric_logger.meters['precision'].update(acc['precision'], n=batch_size)
            metric_logger.meters['recall'].update(acc['recall'], n=batch_size)
            metric_logger.meters['f1_score'].update(acc['f1_score'], n=batch_size)
            # metric_logger.meters['auc_roc'].update(acc['auc_roc'], n=batch_size)
        else:
            metric_logger.meters['precision'].update(acc['macro_precision'], n=batch_size)
            metric_logger.meters['recall'].update(acc['macro_recall'], n=batch_size)
            metric_logger.meters['f1_score'].update(acc['macro_f1_score'], n=batch_size)
            # metric_logger.meters['weighted_f1_score'].update(acc['weighted_f1_score'], n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* precision {precision.global_avg:.3f} recall {recall.global_avg:.3f} f1_score {f1_score.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(precision=metric_logger.precision, recall=metric_logger.recall, f1_score=metric_logger.f1_score, losses=metric_logger.loss))

    # TODO: left for further inferencing
    # if num_classes == 2:
    #     print('* precision {precision.global_avg:.3f} recall {recall.global_avg:.3f} f1_score {f1_score.global_avg:.3f} auc_roc {auc_roc.global_avg:.3f} loss {losses.global_avg:.3f}'
    #           .format(precision=metric_logger.precision, recall=metric_logger.recall, f1_score=metric_logger.f1_score, auc_roc=metric_logger.auc_roc, losses=metric_logger.loss))
    # else:
    #     print('* macro_precision {macro_precision.global_avg:.3f} macro_recall {macro_recall.global_avg:.3f} macro_f1_score {macro_f1_score.global_avg:.3f} weighted_f1_score {weighted_f1_score.global_avg:.3f} loss {losses.global_avg:.3f}'
    #           .format(macro_precision=metric_logger.macro_precision, macro_recall=metric_logger.macro_recall, macro_f1_score=metric_logger.macro_f1_score, weighted_f1_score=metric_logger.weighted_f1_score, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
@torch.no_grad()
def evaluate_eco_multi_model_data(data_loader, model, device, args):
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    predictions = []
    true_values = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        rs_image = batch["rs_image"].to(device, non_blocking=True)
        sv_image = batch["sv_image"].to(device, non_blocking=True)
        sv_location = batch["sv_location"].to(device, non_blocking=True)
        bbox_information = batch["bbox_information"].to(device, non_blocking=True)
        target = batch["values"].to(device, non_blocking=True)
        # print('images and targets')


        # print("before pass model")
        # compute output

        with torch.cuda.amp.autocast():

            output = model(rs_image, sv_image, sv_location, bbox_information)
            loss = criterion(output, target)

        predictions.append(output.detach().cpu())
        true_values.append(target.detach().cpu())

        acc = eval_accuracy_fn.regression_accuracy(output, target)
        # print(acc1, acc5, flush=True)

        batch_size = rs_image.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['MSE'].update(acc['MSE'], n=batch_size)
        metric_logger.meters['RMSE'].update(acc['RMSE'], n=batch_size)
        metric_logger.meters['MAE'].update(acc['MAE'], n=batch_size)
        metric_logger.meters['R2'].update(acc['R2'], n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* MSE@1 {MSE.global_avg:.3f} R2 {R2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(MSE=metric_logger.MSE, R2=metric_logger.R2, losses=metric_logger.loss))

    all_predictions = torch.cat(predictions, dim=0)
    all_true_values = torch.cat(true_values, dim=0)

    indicator_results = eval_accuracy_fn.calculate_and_print_metrics(all_predictions,all_true_values)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, indicator_results
@torch.no_grad()
def evaluate_eco_data(data_loader, model, device, args):
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    predictions = []
    true_values = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["image"]
        target = batch["values"]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output

        with torch.cuda.amp.autocast():
            if args.benchmark == "imagenet" or args.benchmark == "street_view_imagenet":
                output = model(images)
                output = output["pooler_output"][:,0,:]
            elif args.benchmark == "croma":
                output,_ = model(images)
            else:
                output = model(images)
            loss = criterion(output, target)

        predictions.append(output.detach().cpu())
        true_values.append(target.detach().cpu())

        acc = eval_accuracy_fn.regression_accuracy(output, target)
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['MSE'].update(acc['MSE'], n=batch_size)
        metric_logger.meters['RMSE'].update(acc['RMSE'], n=batch_size)
        metric_logger.meters['MAE'].update(acc['MAE'], n=batch_size)
        metric_logger.meters['R2'].update(acc['R2'], n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* MSE@1 {MSE.global_avg:.3f} R2 {R2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(MSE=metric_logger.MSE, R2=metric_logger.R2, losses=metric_logger.loss))

    all_predictions = torch.cat(predictions, dim=0)
    all_true_values = torch.cat(true_values, dim=0)

    indicator_results = eval_accuracy_fn.calculate_and_print_metrics(all_predictions,all_true_values)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, indicator_results
@torch.no_grad()
def evaluate_perception(data_loader, model, device, args):
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    predictions = []
    true_values = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["image"]
        target = batch["values"]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output

        with torch.cuda.amp.autocast():
            if args.benchmark == "imagenet" or args.benchmark == "street_view_imagenet":
                output = model(images)
                output = output["pooler_output"][:,0,:]
            elif args.benchmark == "croma":
                output,_ = model(images)
            else:
                output = model(images)
            loss = criterion(output, target)

        predictions.append(output.detach().cpu())
        true_values.append(target.detach().cpu())

        acc = eval_accuracy_fn.regression_accuracy(output, target)
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['MSE'].update(acc['MSE'], n=batch_size)
        metric_logger.meters['RMSE'].update(acc['RMSE'], n=batch_size)
        metric_logger.meters['MAE'].update(acc['MAE'], n=batch_size)
        metric_logger.meters['R2'].update(acc['R2'], n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* MSE@1 {MSE.global_avg:.3f} R2 {R2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(MSE=metric_logger.MSE, R2=metric_logger.R2, losses=metric_logger.loss))

    all_predictions = torch.cat(predictions, dim=0)
    all_true_values = torch.cat(true_values, dim=0)

    indicator_results = eval_accuracy_fn.calculate_and_print_metrics_perception(all_predictions,all_true_values)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, indicator_results

@torch.no_grad()
def evaluate_fmow_data(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    predictions = []
    true_values = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["image"]
        target = batch["values"]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output

        with torch.cuda.amp.autocast():
            if args.benchmark == "imagenet" or args.benchmark == "street_view_imagenet":
                output = model(images)
                output = output["pooler_output"][:,0,:]
            elif args.benchmark == "croma":
                output,_ = model(images)
            else:
                output = model(images)
            loss = criterion(output, target)

        predictions.append(output.detach().cpu())
        true_values.append(target.detach().cpu())

        acc = eval_accuracy_fn.calculate_classification_metrics(output, target)
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['hit1'].update(acc['hit@1'], n=batch_size)
        metric_logger.meters['hit3'].update(acc['hit@3'], n=batch_size)
        metric_logger.meters['Mean_Reciprocal_Rank'].update(acc['Mean_Reciprocal_Rank'], n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* hit1 {MSE.global_avg:.3f} hit3 {R2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(MSE=metric_logger.hit1, R2=metric_logger.hit3, losses=metric_logger.loss))

    # indicator_results = eval_accuracy_fn.calculate_and_print_metrics(all_predictions,all_true_values)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_fmow_data_for_geobais(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    predictions = []
    true_values = []
    lons = []
    lats = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch["image"]
        target = batch["values"]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output

        with torch.cuda.amp.autocast():
            if args.benchmark == "imagenet" or args.benchmark == "street_view_imagenet":
                output = model(images)
                output = output["pooler_output"][:,0,:]
            elif args.benchmark == "croma":
                output,_ = model(images)
            else:
                output = model(images)

        probs = F.softmax(output, dim=1)
        predictions.extend(probs.detach().cpu())
        true_values.extend(target.detach().cpu())
        lons.extend(batch["lon"].detach().cpu())
        lats.extend(batch["lat"].detach().cpu())
        # 获取每个样本的 top-k 类别和对应概率

    top_k_probs, top_k_classes = torch.topk(probs, 3, dim=1)

    with open("/data/zeping_data/data/croma_worldstrat_all.csv", mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["lon", "lat", "hit@1", "hit@3", "reciprocal_rank", "true_class_probability"])
        writer.writeheader()

        # 遍历每个样本，计算指标并写入 CSV 文件
        for i in range(len(predictions)):
            # 获取该样本的 top-3 类别和对应概率
            top_k_probs, top_k_classes = torch.topk(predictions[i], 3, dim=0)

            # 计算 hit@1
            hit_1 = 1.0 if top_k_classes[0] == true_values[i] else 0.0

            # 计算 hit@3
            hit_3 = 1.0 if true_values[i] in top_k_classes else 0.0

            # 计算 Reciprocal Rank (RR)
            rank = (top_k_classes == true_values[i]).nonzero(as_tuple=False)
            rr = 1.0 / (rank[0].item() + 1) if rank.numel() > 0 else 0.0

            # 计算 true class probability
            true_class_prob = predictions[i][true_values[i]].item()

            # 将结果写入 CSV 文件
            writer.writerow({
                "lon": float(lons[i]),
                "lat": float(lats[i]),
                "hit@1": hit_1,
                "hit@3": hit_3,
                "reciprocal_rank": rr,
                "true_class_probability": true_class_prob
            })

    print("结果已写入 ")
    return 0




@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_temporal(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    tta = False

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        timestamps = batch[1]
        target = batch[-1]

        batch_size = images.shape[0]
        # print(images.shape, timestamps.shape, target.shape)
        if tta:
            images = images.reshape(-1, 3, 3, 224, 224)
            timestamps = timestamps.reshape(-1, 3, 3)
            target = target.reshape(-1, 1)
        # images = images.reshape()
        # print('images and targets')
        images = images.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, timestamps)

            if tta:
                # output = output.reshape(batch_size, 9, -1).mean(dim=1, keepdims=False)

                output = output.reshape(batch_size, 9, -1)
                sp = output.shape
                maxarg = output.argmax(dim=-1)

                output = F.one_hot(maxarg.reshape(-1), num_classes=1000).float()
                output = output.reshape(sp).mean(dim=1, keepdims=False)
                # print(output.shape)
                
                target = target.reshape(batch_size, 9)[:, 0]
            # print(target.shape)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
