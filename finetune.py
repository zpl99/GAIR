# coding:utf-8
"""
@file: finetune_pipeline.py
@author: Zeping Liu
@ide: PyCharm
@createTime: 2024.08
@contactInformation: zeping.liu@utexas.edu
@Function: for model finetune and deploy to downstream tasks
"""
import argparse
import datetime
import json
import numpy as np
import os
import time

import model_cl_RS_sc
import wandb
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch import nn

import timm

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from transformers import ViTImageProcessor, ViTModel

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_fmow_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit
from croma import use_croma
from engine_finetune import (train_one_epoch, train_one_epoch_temporal, train_one_epoch_eco_data,
                             evaluate, evaluate_temporal, evaluate_eco_data, evaluate_perception)
from transformers import ViTImageProcessor, ViTModel, ViTConfig


def get_args_parser():
    parser = argparse.ArgumentParser('GAIR fine-tuning for downstream tasks', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--basemodel', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=96, type=int,
                        help='images input size')
    parser.add_argument('--input_sv_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='images input size')
    parser.add_argument('--finetune_type', type=str, default="finetune_adapter",
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Finetuning params
    parser.add_argument('--finetune', default='/data/zeping_data/data/gsv_rs_project/nerf_style_pretrained_checkpoint/nerf_liif_2m_with_loc_with_distillation/checkpoint-155.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--benchmark', type=str, default="street_view_liif")
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    parser.add_argument('--train_path', default="/data/zeping_data/data/gsv_rs_project/street_scapes/perception_task/train_perception.csv",
                        type=str,
                        help='Train .csv path')
    parser.add_argument('--test_path', default="/data/zeping_data/data/gsv_rs_project/street_scapes/perception_task/test_perception.csv",
                        type=str,
                        help='Test .csv path')

    parser.add_argument('--dataset_type', default='perception',
                        help='Whether to use fmow rgb, sentinel, or other dataset.')

    parser.add_argument('--nb_classes', default=62, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir',
                        default='/data/zeping_data/data/gsv_rs_project/perception/temp', # streetview_liif_pos_finetune_1m
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir',
                        default='/data/zeping_data/data/gsv_rs_project/streetview_liif_finetune_log',
                        help='path where to tensorboard log')

    parser.add_argument('--multi_model', default=False,
                        help='whether to use multi modality for finetuning')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')


    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--save_every', type=int, default=1, help='How frequently (in epochs) to save ckpt')
    parser.add_argument('--wandb', type=str, default="sat_sv_finetune_perception",
                        help="Wandb project name, eg: sentinel_finetune")
    # parser.add_argument('--wandb', type=str, default=None,
    #                     help="Wandb project name, eg: sentinel_finetune")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def load_street_view_liif_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    temp_model = model_cl_RS_sc.vit_base_patch16_dec512d8b_liif(rs_input_size=96, sv_input_size=224)
    temp_model.load_state_dict(checkpoint['model'])
    street_view_checkpoint = temp_model.sv_encoder.state_dict()


    model.load_state_dict(street_view_checkpoint, strict=False)
    if args.finetune_type == "finetune_non_linear":
        model.head = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 6))
    return model


def initialize_sequential_weights(sequential, init_fn, **kwargs):
    for module in sequential:
        if hasattr(module, 'weight'):  # check if the module has a weight attribute
            init_fn(module.weight, **kwargs)

def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    dataset_train = build_fmow_dataset(is_train=True, args=args)
    dataset_val = build_fmow_dataset(is_train=False, args=args)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank,
            shuffle=True)  # shuffle=True to reduce monitor bias

    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )



    if args.benchmark == "street_view_liif":
        config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_labels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            layer_norm_eps=1e-12,
            dropout_rate=0.1,
            attention_probs_dropout_prob=0.1
        )


        # from vit model to add a head for classification
        class ViTWithHead(ViTModel):
            def __init__(self, config, num_labels):
                super(ViTWithHead, self).__init__(config)
                self.head = nn.Linear(config.hidden_size, num_labels)

            def forward(self, pixel_values):
                outputs = super().forward(pixel_values)
                pooler_output = outputs.pooler_output  # [batch_size, hidden_size]
                logits = self.head(pooler_output)
                return logits

        base_model = ViTWithHead(config, num_labels=6)
        load_street_view_liif_checkpoint(base_model,args.finetune)
    else:
        raise NotImplementedError

    base_model.to(device)
    model_without_ddp = base_model

    if args.finetune_type == "frozen":
        print("frozen")
        for name, params in base_model.named_parameters():
            params.requires_grad = False
            print(f"{name} won't require gradient")
        test_stats = evaluate_eco_data(data_loader_val, base_model, device)
        print(f"Evaluation on {len(dataset_val)} test images- MSE: {test_stats['MSE']:.2f}%, "
              f"R2: {test_stats['R2']:.2f}%")
        exit(0)
    elif args.finetune_type == "finetune_all":
        print("finetune_all")
    elif args.finetune_type in ["finetune_adapter", "finetune_non_linear"]:
        print("finetune_adapter")
        for name, params in base_model.named_parameters():
            if "head" in name or "pooler" in name:  # only tune head
                params.requires_grad = True
                print(f"{name} will require gradient")
            else:
                params.requires_grad = False
                print(f"{name} won't require gradient")

    else:
        raise NotImplementedError

    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)



    model = base_model



    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()
    criterion = nn.MSELoss()

    print("criterion = %s" % str(criterion))

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        print(args.wandb)
        wandb.init(project=args.wandb, name=args.benchmark+args.finetune_type)
        wandb.config.update(args)
        wandb.watch(model)

    if args.eval:
        test_stats = evaluate_perception(data_loader_val, model, device, args)
        print(f"Evaluation on {len(dataset_val)} test images- MSE: {test_stats['MSE']:.2f}%, R2: {test_stats['R2']:.2f}%, RMSE: {test_stats['RMSE']:.2f}, MAE: {test_stats['MAE']:.2f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_eco_data(base_model, criterion, data_loader_train, optimizer, device, epoch,
                                               loss_scaler,  # do not use mixup_fn
                                               args.clip_grad,
                                               log_writer=log_writer,
                                               args=args)

        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=base_model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats,indicator_results = evaluate_perception(data_loader_val, base_model, device, args)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['MSE']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["MSE"])  # utilize MSE as the major metric
        print(f'Max accuracy (MSE): {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_MSE', test_stats['MSE'], epoch)
            log_writer.add_scalar('perf/test_R2', test_stats['R2'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': float(v) for k, v in train_stats.items()},
                     **{f'test_{k}': float(v) for k, v in test_stats.items()},
                     'epoch': float(epoch),
                     'n_parameters': float(n_parameters)}

        for indicator, metrics in indicator_results.items():
            for metric_name, metric_value in metrics.items():
                log_stats[f'{indicator}/{metric_name}_test'] = float(metric_value)

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.wandb is not None:
                try:
                    wandb.log(log_stats)
                except ValueError:
                    print(f"Invalid stats?")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    import os
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(args.train_path)
    main(args)
