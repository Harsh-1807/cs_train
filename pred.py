'''
SemDINO Prediction / Visualization Script
Supports: Landsat-SCD, SECOND, HRSCD datasets
Windows-compatible, single GPU or CPU

Produces exactly 3 outputs per image pair, nothing else:
  - im1/<name>.png     colorized semantic segmentation for time-1, changed regions only
  - im2/<name>.png     colorized semantic segmentation for time-2, changed regions only
  - change/<name>.png  binary change mask (0/255)
'''
import os
import time
import argparse
import numpy as np
import torch
from skimage import io
from torch.nn import functional as F
from torch.utils.data import DataLoader


def get_device():
    if torch.cuda.is_available():
        print(f'[INFO] Using GPU: {torch.cuda.get_device_name(0)}')
        return torch.device('cuda')
    else:
        print('[INFO] CUDA not available, using CPU.')
        return torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(
        description='SemDINO Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='SECOND',
                        choices=['Landsat', 'SECOND', 'HRSCD', 'CustomCD'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory of dataset')
    parser.add_argument('--pred_batch_size', type=int, default=1,
                        help='Prediction batch size')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Directory containing test images (with im1/ and im2/ subfolders)')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory to save prediction outputs')
    parser.add_argument('--chkpt_path', type=str, required=True,
                        help='Path to trained model checkpoint .pth file')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to DINOv3 pretrained weights .pth file')
    parser.add_argument('--efficientnet_weights_path', type=str, default=None,
                        help='Path to a local EfficientNet-B3 ImageNet .pth file '
                             '(unnecessary at inference time if --chkpt_path already '
                             'contains trained backbone weights, but harmless to pass)')
    parser.add_argument('--flip', action='store_true',
                        help='Enable test-time augmentation with flips')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers (default: 0 for Windows)')
    parser.add_argument('--dim', type=int, default=128,
                        help='Feature dimension (default: 128)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    begin_time = time.time()

    if args.dataset == 'Landsat':
        from datasets import RS_Landsat as RS
    elif args.dataset == 'SECOND':
        from datasets import RS_SECOND as RS
    elif args.dataset == 'HRSCD':
        from datasets import RS_HRSCD as RS
    elif args.dataset == 'CustomCD':
        from datasets import RS_CUSTOM as RS

    if args.data_root is not None:
        RS.root = args.data_root

    from SemDINO import SemDINO
    # The checkpoint below overwrites every weight anyway; efficientnet_weights_path
    # is accepted here only in case a partial/legacy checkpoint ever needs a base.
    net = SemDINO(num_classes=RS.num_classes, dim=args.dim,
                  weights_path=args.weights_path,
                  efficientnet_weights_path=args.efficientnet_weights_path).to(device)

    ckpt = torch.load(args.chkpt_path, map_location='cpu')
    state = ckpt['model'] if (isinstance(ckpt, dict) and 'model' in ckpt) else ckpt
    new_state = {}
    for key, val in state.items():
        if key.startswith("module."):
            new_state[key[7:]] = val
        else:
            new_state[key] = val
    net.load_state_dict(new_state)
    net.eval()

    test_set = RS.Data_test(args.test_dir)
    test_loader = DataLoader(test_set, batch_size=args.pred_batch_size,
                             num_workers=args.num_workers)
    predict(net, test_set, test_loader, args.pred_dir, RS, device, flip=args.flip)
    print('Total time: %.2fs' % (time.time() - begin_time))


def predict(net, pred_set, pred_loader, pred_dir, RS, device, flip=False):
    """
    Model has 4 outputs: change, semA, semB, edge.
    Only 3 are ever written to disk: colorized semA, colorized semB (both
    masked to changed regions only), and the binary change mask. `edge` is
    a training-time auxiliary signal only and is never saved here.
    """
    im1_dir = os.path.join(pred_dir, 'im1')
    im2_dir = os.path.join(pred_dir, 'im2')
    change_dir = os.path.join(pred_dir, 'change')
    os.makedirs(im1_dir, exist_ok=True)
    os.makedirs(im2_dir, exist_ok=True)
    os.makedirs(change_dir, exist_ok=True)

    batch_size = pred_loader.batch_size

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B = data
        imgs_A = imgs_A.to(device).float()
        imgs_B = imgs_B.to(device).float()
        curr_batch = imgs_A.shape[0]

        with torch.no_grad():
            out_change, outputs_A, outputs_B, _ = net(imgs_A, imgs_B)
            out_change = torch.sigmoid(out_change)

        if flip:
            outputs_A = F.softmax(outputs_A, dim=1)
            outputs_B = F.softmax(outputs_B, dim=1)

            # Vertical flip
            imgs_A_v = torch.flip(imgs_A, [2])
            imgs_B_v = torch.flip(imgs_B, [2])
            with torch.no_grad():
                out_change_v, outputs_A_v, outputs_B_v, _ = net(imgs_A_v, imgs_B_v)
            outputs_A_v = torch.flip(outputs_A_v, [2])
            outputs_B_v = torch.flip(outputs_B_v, [2])
            out_change_v = torch.flip(torch.sigmoid(out_change_v), [2])
            outputs_A += F.softmax(outputs_A_v, dim=1)
            outputs_B += F.softmax(outputs_B_v, dim=1)
            out_change += out_change_v

            # Horizontal flip
            imgs_A_h = torch.flip(imgs_A, [3])
            imgs_B_h = torch.flip(imgs_B, [3])
            with torch.no_grad():
                out_change_h, outputs_A_h, outputs_B_h, _ = net(imgs_A_h, imgs_B_h)
            outputs_A_h = torch.flip(outputs_A_h, [3])
            outputs_B_h = torch.flip(outputs_B_h, [3])
            out_change_h = torch.flip(torch.sigmoid(out_change_h), [3])
            outputs_A += F.softmax(outputs_A_h, dim=1)
            outputs_B += F.softmax(outputs_B_h, dim=1)
            out_change += out_change_h

            # Both flips
            imgs_A_hv = torch.flip(imgs_A, [2, 3])
            imgs_B_hv = torch.flip(imgs_B, [2, 3])
            with torch.no_grad():
                out_change_hv, outputs_A_hv, outputs_B_hv, _ = net(imgs_A_hv, imgs_B_hv)
            outputs_A_hv = torch.flip(outputs_A_hv, [2, 3])
            outputs_B_hv = torch.flip(outputs_B_hv, [2, 3])
            out_change_hv = torch.flip(torch.sigmoid(out_change_hv), [2, 3])
            outputs_A += F.softmax(outputs_A_hv, dim=1)
            outputs_B += F.softmax(outputs_B_hv, dim=1)
            out_change += out_change_hv
            out_change = out_change / 4

        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = (out_change.cpu().detach() > 0.5).squeeze(1)  # [B, H, W] bool
        preds_A = torch.argmax(outputs_A, dim=1)  # [B, H, W]
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A * change_mask.long()).numpy()
        preds_B = (preds_B * change_mask.long()).numpy()
        change_np = (change_mask.numpy() * 255).astype(np.uint8)

        for j in range(curr_batch):
            sample_idx = vi * batch_size + j
            mask_name = pred_set.get_mask_name(sample_idx)

            io.imsave(os.path.join(im1_dir, mask_name), RS.Index2Color(preds_A[j]))
            io.imsave(os.path.join(im2_dir, mask_name), RS.Index2Color(preds_B[j]))
            io.imsave(os.path.join(change_dir, mask_name), change_np[j])

        print(f'[{vi * batch_size + curr_batch}/{len(pred_set)}] saved')


if __name__ == '__main__':
    main()
