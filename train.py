'''
SemDINO Training Script (multi-GPU, torchrun/DDP)

Supports: Landsat-SCD, SECOND, HRSCD.
Launch with torchrun for multi-GPU, e.g.:

    torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
        --dataset SECOND --data_root /path/to/SECOND --batch_size 8 --amp

Falls back to single-process CPU/GPU if launched with plain `python train.py`
(no RANK/WORLD_SIZE env vars set by torchrun).
'''
import os
os.environ.setdefault('TMPDIR', '/tmp')
os.makedirs('/tmp', exist_ok=True)
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.autograd
from torch import optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity
from utils.utils import accuracy, SCDD_eval_all, AverageMeter

# module path + num_eval_classes per dataset.
# NOTE: HRSCD's 26 is carried over unchanged from the original script - it does
# not match the 6-class description in details.md. If you rely on HRSCD, verify
# this against utils/utils.py::SCDD_eval_all before trusting the Fscd/Sek numbers.
# SECOND and Landsat default to num_classes from the dataset module itself.
DATASET_EVAL_CLASSES = {
    'HRSCD': 26,
}


def setup_distributed():
    """Initialize torch.distributed if launched via torchrun, else run single-process."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return rank, world_size, local_rank, device, True
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f'[INFO] Single-process mode. Using GPU: {torch.cuda.get_device_name(0)}')
        else:
            device = torch.device('cpu')
            print('[INFO] Single-process mode. CUDA not available, using CPU.')
        return 0, 1, 0, device, False


def unwrap_model(net):
    """Return the underlying nn.Module whether wrapped in DDP/DataParallel or not."""
    return net.module if isinstance(net, (nn.DataParallel, DDP)) else net


def set_seed(seed, rank=0):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='SemDINO Training (DDP)')
    parser.add_argument('--dataset', type=str, default='SECOND',
                        choices=['Landsat', 'SECOND', 'HRSCD'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory of dataset (must contain train/ and val/)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-GPU training batch size')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Per-GPU validation batch size (validation runs on rank 0 only)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr_decay_power', type=float, default=1.5,
                        help='Polynomial LR decay power')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                        help='Max grad norm for clipping (<=0 disables)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader num_workers per process')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='Print frequency in iterations')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to DINOv3 pretrained weights .pth file')
    parser.add_argument('--efficientnet_weights_path', type=str, default=None,
                        help='Path to a local torchvision EfficientNet-B3 ImageNet '
                             '.pth file. Optional: falls back to EFFICIENTNET_B3_WEIGHTS '
                             'env var, then $TORCH_HOME cache, then random init with a '
                             'warning. Never downloads anything (safe for offline HPC).')
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='Enable gradient checkpointing in the EfficientNet backbone '
                             '(trades compute for memory -- use if you hit OOM before '
                             'lowering batch_size)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint dict to resume training from')
    parser.add_argument('--dim', type=int, default=128,
                        help='Feature dimension')
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision (recommended on H100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (offset by rank)')
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='Pass find_unused_parameters=True to DDP (slower; only enable if '
                             'you hit an "unused parameter" runtime error)')
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank, device, distributed = setup_distributed()
    is_main = (rank == 0)
    set_seed(args.seed, rank)
    torch.backends.cudnn.benchmark = True

    if args.dataset == 'Landsat':
        from datasets import RS_Landsat as RS
    elif args.dataset == 'SECOND':
        from datasets import RS_SECOND as RS
    elif args.dataset == 'HRSCD':
        from datasets import RS_HRSCD as RS
    num_eval_classes = DATASET_EVAL_CLASSES.get(args.dataset, RS.num_classes)
    NET_NAME = 'SemDINO'
    DATA_NAME = args.dataset

    if args.data_root is not None:
        RS.root = args.data_root

    chkpt_dir = os.path.join(working_path, 'checkpoint', DATA_NAME, NET_NAME)
    log_dir = os.path.join(working_path, 'logs', DATA_NAME, NET_NAME)
    writer = None
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(chkpt_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f'[INFO] world_size={world_size} distributed={distributed} '
              f'dataset={DATA_NAME} num_classes={RS.num_classes} num_eval_classes={num_eval_classes}')

    # Build model
    from SemDINO import SemDINO
    net = SemDINO(num_classes=RS.num_classes, dim=args.dim,
                  weights_path=args.weights_path,
                  efficientnet_weights_path=args.efficientnet_weights_path,
                  use_checkpoint=args.use_checkpoint)

    freeze_model(net.encoder.dino)  # type: ignore

    start_epoch = 0
    best_state = {'bestFscdV': 0.0, 'bestaccV': 0.0, 'bestloss': 1.0, 'bestaccT': 0.0}

    # Resume BEFORE wrapping in DDP/DataParallel so the saved (unwrapped) state
    # dict loads cleanly regardless of world size at save vs. load time.
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        if 'model' in ckpt:
            net.load_state_dict(ckpt['model'])
            start_epoch = ckpt.get('epoch', 0)
            best_state.update(ckpt.get('best_state', best_state))
            if is_main:
                print(f'[INFO] Resumed from {args.resume} at epoch {start_epoch}')
        else:
            # Backward-compatible: a bare state_dict (old-style checkpoint)
            net.load_state_dict(ckpt)
            if is_main:
                print(f'[INFO] Resumed weights only (no epoch/optimizer state) from {args.resume}')

    net = net.to(device)

    if distributed:
        net = DDP(net, device_ids=[local_rank], output_device=local_rank,
                  find_unused_parameters=args.find_unused_parameters)
    elif torch.cuda.device_count() > 1:
        # Fallback if someone runs this without torchrun on a multi-GPU box.
        print('[WARN] Multiple GPUs visible but not launched via torchrun; '
              'falling back to DataParallel. Prefer torchrun for real training.')
        net = nn.DataParallel(net)

    # Data loaders
    train_set = RS.Data('train', random_flip=True)
    val_set = RS.Data('val') if is_main else None  # validation runs on rank 0 only

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank,
                                       shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=train_sampler, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=(args.num_workers > 0), drop_last=True)
    val_loader = None
    if is_main:
        val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True,
                                persistent_workers=(args.num_workers > 0))

    # Loss and optimizer
    criterion = CrossEntropyLoss2d(ignore_index=0).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=args.lr, weight_decay=args.weight_decay,
                          momentum=args.momentum, nesterov=True)
    scaler = GradScaler(enabled=args.amp)

    train(train_loader, train_sampler, net, criterion, optimizer, scaler, val_loader,
          writer, args, device, chkpt_dir, NET_NAME, RS, num_eval_classes,
          start_epoch, best_state, rank, is_main, distributed)

    if is_main and writer is not None:
        writer.close()
        print('Training finished.')
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def train(train_loader, train_sampler, net, criterion, optimizer, scaler, val_loader,
          writer, args, device, chkpt_dir, NET_NAME, RS, num_eval_classes,
          start_epoch, best_state, rank, is_main, distributed):
    bestaccT = best_state['bestaccT']
    bestFscdV = best_state['bestFscdV']
    bestloss = best_state['bestloss']
    bestaccV = best_state['bestaccV']
    begin_time = time.time()
    all_iters = float(len(train_loader) * args.epochs)
    criterion_sc = ChangeSimilarity().to(device)
    curr_epoch = start_epoch

    best_ckpt_path = None
    latest_ckpt_path = None

    while True:
        net.train()
        if train_sampler is not None:
            train_sampler.set_epoch(curr_epoch)
        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters, args.lr, args.lr_decay_power)
            imgs_A, imgs_B, labels_A, labels_B = data

            imgs_A = imgs_A.to(device, non_blocking=True).float()
            imgs_B = imgs_B.to(device, non_blocking=True).float()
            labels_bn = (labels_A > 0).unsqueeze(1).to(device, non_blocking=True).float()
            labels_A = labels_A.to(device, non_blocking=True).long()
            labels_B = labels_B.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                out_change, outputs_A, outputs_B, out_edge = net(imgs_A, imgs_B)
                assert outputs_A.size()[1] == RS.num_classes
                loss_seg = criterion(outputs_A, labels_A) * 0.5 + criterion(outputs_B, labels_B) * 0.5
                loss_bn = weighted_BCE_logits(out_change, labels_bn)
                loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
                loss_edge = F.binary_cross_entropy_with_logits(out_edge, labels_bn)
                loss = loss_seg + loss_bn + loss_sc + loss_edge * 0.1

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = torch.sigmoid(out_change).cpu().detach() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A * change_mask.squeeze(1).long()).numpy()
            preds_B = (preds_B * change_mask.squeeze(1).long()).numpy()
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            curr_time = time.time() - start

            if is_main and (i + 1) % args.print_freq == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, acc_meter.val * 100))  # type: ignore
                writer.add_scalar('train seg_loss', train_seg_loss.val, running_iter)
                writer.add_scalar('train bn_loss', train_bn_loss.val, running_iter)
                writer.add_scalar('train sc_loss', train_sc_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        # Validation runs only on rank 0 (full val set, no sampler) so the
        # Fscd/mIoU/Sek numbers aren't skewed by sharding across ranks.
        if is_main:
            Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(
                val_loader, net, criterion, curr_epoch, writer, device, num_eval_classes, args.amp)
        else:
            Fscd_v = mIoU_v = Sek_v = acc_v = loss_v = 0.0

        if distributed:
            dist.barrier()

        if is_main:
            if acc_meter.avg > bestaccT:
                bestaccT = acc_meter.avg  # type: ignore

            state_dict_to_save = unwrap_model(net).state_dict()
            checkpoint_payload = {
                'model': state_dict_to_save,
                'epoch': curr_epoch + 1,
                'best_state': {
                    'bestaccT': bestaccT, 'bestFscdV': bestFscdV,
                    'bestloss': bestloss, 'bestaccV': bestaccV,
                },
            }

            # 1. Save latest checkpoint
            new_latest_path = os.path.join(chkpt_dir, f"{NET_NAME}_latest_epoch{curr_epoch}.pth")
            torch.save(checkpoint_payload, new_latest_path)
            if latest_ckpt_path is not None and os.path.exists(latest_ckpt_path):
                os.remove(latest_ckpt_path)
            latest_ckpt_path = new_latest_path

            # 2. Save best checkpoint (based on Fscd)
            if Fscd_v > bestFscdV:
                bestFscdV = Fscd_v
                bestaccV = acc_v
                bestloss = loss_v
                checkpoint_payload['best_state']['bestFscdV'] = bestFscdV
                checkpoint_payload['best_state']['bestaccV'] = bestaccV
                checkpoint_payload['best_state']['bestloss'] = bestloss

                new_best_path = os.path.join(
                    chkpt_dir, f"{NET_NAME}_best_epoch{curr_epoch}_Fscd{Fscd_v*100:.2f}.pth")
                torch.save(checkpoint_payload, new_best_path)
                if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                    os.remove(best_ckpt_path)
                best_ckpt_path = new_best_path
                print(f'--> Saved new best checkpoint with Fscd: {bestFscdV * 100:.2f}')

            print('Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2f acc %.2f loss %.4f'
                  % (time.time() - begin_time, bestaccT * 100, bestFscdV * 100, bestaccV * 100, bestloss))  # type: ignore

        curr_epoch += 1
        if curr_epoch >= args.epochs:
            return


def validate(val_loader, net, criterion, curr_epoch, writer, device, num_eval_classes, amp):
    net.eval()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data

        imgs_A = imgs_A.to(device, non_blocking=True).float()
        imgs_B = imgs_B.to(device, non_blocking=True).float()
        labels_A = labels_A.to(device, non_blocking=True).long()
        labels_B = labels_B.to(device, non_blocking=True).long()

        with torch.no_grad(), autocast(enabled=amp):
            out_change, outputs_A, outputs_B, out_edge = net(imgs_A, imgs_B)
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = torch.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.squeeze(1).long()).numpy()
        preds_B = (preds_B * change_mask.squeeze(1).long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, num_eval_classes)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'
          % (curr_time, val_loss.average(), Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))  # type: ignore

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr, lr_decay_power):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** lr_decay_power)
    running_lr = init_lr * scale_running_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
