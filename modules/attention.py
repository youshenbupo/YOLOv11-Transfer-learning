"""
Attention mechanisms for YOLO11-UDA.

CBAM (Convolutional Block Attention Module) and SE (Squeeze-and-Excitation)
can be inserted into YOLO11 backbone to enhance feature representation
for domain adaptation.

Reference:
    - CBAM: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
    - SE: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention using global average and max pooling."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        # Global average pool
        avg_out = self.fc(x.mean(dim=[2, 3]))
        # Global max pool
        max_out = self.fc(x.amax(dim=[2, 3]))
        att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return att


class SpatialAttention(nn.Module):
    """Spatial attention using channel-wise statistics."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(combined))
        return att


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Applies channel attention followed by spatial attention.
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Channel attention only (no spatial attention).
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        att = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * att


def insert_cbam_after_block(model, block_indices, reduction=16):
    """
    Insert CBAM modules after specified blocks in a YOLO11 model.

    Args:
        model: YOLO11 nn.Module (model.model is nn.Sequential)
        block_indices: list of int, block indices to insert CBAM after
        reduction: channel reduction ratio for CBAM

    Returns:
        model: Modified model with CBAM modules inserted
    """
    import torch.nn as nn

    blocks = list(model.model)
    offset = 0

    for idx in sorted(block_indices):
        actual_idx = idx + offset
        # Get output channels from the block
        block = blocks[actual_idx]
        out_channels = _get_output_channels(block)
        if out_channels is None:
            print(f"[CBAM] Warning: Cannot determine output channels for block {idx}, skipping.")
            continue

        cbam = CBAM(out_channels, reduction=reduction)
        blocks.insert(actual_idx + 1, cbam)
        offset += 1
        print(f"[CBAM] Inserted after block {idx} (channels={out_channels})")

    model.model = nn.Sequential(*blocks)
    return model


def _get_output_channels(block):
    """Try to determine the output channels of a block."""
    # Check common attributes
    if hasattr(block, "out_channels"):
        return block.out_channels
    if hasattr(block, "cv2"):
        # C3k2 or similar blocks
        if hasattr(block.cv2, "out_channels"):
            return block.cv2.out_channels
    if hasattr(block, "conv"):
        if hasattr(block.conv, "out_channels"):
            return block.conv.out_channels
    # Try last conv in sequential
    for name in reversed(list(block._modules.keys())):
        m = getattr(block, name)
        if isinstance(m, nn.Conv2d):
            return m.out_channels
        if hasattr(m, "out_channels"):
            return m.out_channels
    return None
