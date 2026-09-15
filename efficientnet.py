"""
backbone/efficientnet.py

EfficientNet-B3 feature-pyramid backbone for SemDINO.

Built on torchvision's `efficientnet_b3` architecture. Returns a 4-scale
feature pyramid (strides 4, 8, 16, 32 relative to the input) with channel
counts (32, 48, 136, 384) -- this exactly matches the FPN input channels
declared in SemDINO.py:

    self.fpn = FPN([32, 48, 136, 384], dim)

Offline-safe by design: this module NEVER calls torchvision's weight-download
path (`weights=EfficientNet_B3_Weights.IMAGENET1K_V1`), which would try to
reach the internet and hang/fail on a compute node with no network access.
Instead, `pretrained=True` only loads weights from a local .pth file, checked
in this order:
    1. explicit `weights_path` argument
    2. EFFICIENTNET_B3_WEIGHTS environment variable
    3. $TORCH_HOME/hub/checkpoints/efficientnet_b3_rwightman-b3899882.pth
       (the standard torchvision cache location -- if you download the
       official ImageNet weights on any machine with internet and copy
       the .pth file to that path on the HPC, this finds it automatically)
If none of those exist, the backbone falls back to random initialization
with a printed warning -- it will not silently pretend to be pretrained,
and it will not hang trying to fetch anything.

To get the official ImageNet weights onto an offline cluster:
    # on a machine WITH internet:
    python -c "from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights; \
               efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)"
    # this downloads to ~/.cache/torch/hub/checkpoints/efficientnet_b3_rwightman-b3899882.pth
    # scp that one file to the same relative path (or your $TORCH_HOME) on the HPC.
"""
import os
import warnings

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

try:
    from torchvision.models import efficientnet_b3
except ImportError as e:
    raise ImportError(
        "backbone.efficientnet requires torchvision. Install it with "
        "`pip install torchvision` (matching your torch/CUDA version) before running."
    ) from e


# --------------------------------------------------------------------------
# EfficientNet-B3 stage layout (torchvision's `.features` Sequential has 9
# entries: stem, 7 MBConv stage groups, final 1x1 conv). Stage output
# channels below are B3's width-scaled values (width_mult=1.2, round-to-8),
# cumulative stride is stem(x2) then each stage's own first-block stride.
#
#   features[0] : stem                  stride 2,  channels  40
#   features[1] : stage1 (MBConv e=1)   stride 2,  channels  24
#   features[2] : stage2 (MBConv e=6)   stride 4,  channels  32   <- C2 (kept)
#   features[3] : stage3 (MBConv e=6)   stride 8,  channels  48   <- C3 (kept)
#   features[4] : stage4 (MBConv e=6)   stride 16, channels  96
#   features[5] : stage5 (MBConv e=6)   stride 16, channels 136   <- C4 (kept)
#   features[6] : stage6 (MBConv e=6)   stride 32, channels 232
#   features[7] : stage7 (MBConv e=6)   stride 32, channels 384   <- C5 (kept)
#   features[8] : final 1x1 conv to 1536 -- not used, we stop before it
#
# So the 4 kept scales are exactly the stage indices {2, 3, 5, 7} in the
# original `.features` numbering, giving channels (32, 48, 136, 384).
# --------------------------------------------------------------------------
_KEEP_FEATURE_INDICES = (2, 3, 5, 7)
EXPECTED_CHANNELS = (32, 48, 136, 384)
EXPECTED_STRIDES = (4, 8, 16, 32)


class EfficientNetB3Backbone(nn.Module):
    def __init__(self, pretrained: bool = True, weights_path: str = None,
                use_checkpoint: bool = False):
        super().__init__()
        base = efficientnet_b3(weights=None)  # architecture only, zero network calls
        self.stem = base.features[0]
        self.stages = nn.ModuleList(base.features[1:8])  # original features[1..7]
        self.use_checkpoint = use_checkpoint

        if pretrained:
            self._load_pretrained(weights_path)

    def _load_pretrained(self, weights_path):
        resolved = _resolve_weights_path(weights_path)
        if resolved is None:
            warnings.warn(
                "[EfficientNetB3Backbone] pretrained=True but no local weights file "
                "was found (checked weights_path arg / EFFICIENTNET_B3_WEIGHTS env var "
                "/ $TORCH_HOME cache). Continuing with RANDOM initialization -- see the "
                "module docstring in backbone/efficientnet.py for how to stage the "
                "official weights on an offline cluster.",
                RuntimeWarning,
            )
            return

        state = torch.load(resolved, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']

        # Remap torchvision's "features.<i>...." keys onto this module's
        # "stem...." / "stages.<i-1>...." naming.
        remapped = {}
        for k, v in state.items():
            if not k.startswith('features.'):
                continue  # skip classifier head etc. -- we don't use it
            parts = k.split('.')
            idx = int(parts[1])
            if idx == 0:
                new_key = '.'.join(['stem'] + parts[2:])
            elif 1 <= idx <= 7:
                new_key = '.'.join([f'stages.{idx - 1}'] + parts[2:])
            else:
                continue  # features.8, the final 1x1 conv -- not part of this backbone
            remapped[new_key] = v

        result = self.load_state_dict(remapped, strict=False)
        missing = result.missing_keys
        unexpected = result.unexpected_keys
        print(f'[EfficientNetB3Backbone] Loaded pretrained weights from {resolved} '
              f'({len(remapped)} tensors matched, {len(missing)} missing, '
              f'{len(unexpected)} unexpected).')
        if len(remapped) == 0:
            warnings.warn(
                '[EfficientNetB3Backbone] Weight file was found and loaded but 0 '
                'tensors matched -- the checkpoint format is probably not a plain '
                'torchvision efficientnet_b3 state_dict. Check the file.',
                RuntimeWarning,
            )

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, stage in enumerate(self.stages):
            feature_index = i + 1  # this stage's index in the original `.features`
            if self.use_checkpoint and self.training:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
            if feature_index in _KEEP_FEATURE_INDICES:
                outs.append(x)
        assert len(outs) == 4, f'expected 4 pyramid outputs, got {len(outs)}'
        return outs  # [C2, C3, C4, C5], channels (32, 48, 136, 384)


def _resolve_weights_path(weights_path):
    """Find a local EfficientNet-B3 ImageNet checkpoint without ever touching
    the network. Returns None if nothing is found (caller falls back to
    random init rather than raising, since training can still proceed)."""
    candidates = []
    if weights_path:
        candidates.append(weights_path)
    env_path = os.environ.get('EFFICIENTNET_B3_WEIGHTS')
    if env_path:
        candidates.append(env_path)

    torch_home = os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
    candidates.append(os.path.join(torch_home, 'hub', 'checkpoints',
                                   'efficientnet_b3_rwightman-b3899882.pth'))

    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def efficientnet_b3_backbone(pretrained: bool = True, weights_path: str = None,
                             use_checkpoint: bool = False):
    """
    Factory function. Matches the call already used in SemDINO.py:
        efficientnet_b3_backbone(pretrained=True)
    `weights_path` and `use_checkpoint` are new, optional keyword args --
    SemDINO.py's Encoder has been updated to accept and forward them.
    """
    return EfficientNetB3Backbone(pretrained=pretrained, weights_path=weights_path,
                                  use_checkpoint=use_checkpoint)


if __name__ == '__main__':
    # Standalone sanity check -- run this once (CPU is fine, no GPU needed)
    # before trusting the real training job:
    #     python backbone/efficientnet.py
    print('Building EfficientNetB3Backbone(pretrained=False) for a shape check...')
    net = efficientnet_b3_backbone(pretrained=False)
    net.eval()
    x = torch.randn(1, 3, 416, 416)
    with torch.no_grad():
        feats = net(x)
    ok = True
    for f, exp_ch, exp_stride in zip(feats, EXPECTED_CHANNELS, EXPECTED_STRIDES):
        exp_hw = 416 // exp_stride
        got_ch = f.shape[1]
        got_hw = f.shape[2]
        status = 'OK' if (got_ch == exp_ch and got_hw == exp_hw) else 'MISMATCH'
        if status != 'OK':
            ok = False
        print(f'  stride {exp_stride:>2}: expected ch={exp_ch}, hw={exp_hw:>3}  |  '
              f'got ch={got_ch}, hw={got_hw:>3}  [{status}]')
    print('ALL SHAPES OK -- safe to wire into FPN([32,48,136,384], dim)' if ok
          else 'SHAPE MISMATCH -- do not run the real training job yet')
