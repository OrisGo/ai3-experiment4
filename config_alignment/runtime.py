"""Shared precision selection for CUDA training and evaluation."""

import torch


def compute_dtype() -> torch.dtype:
    """Prefer native BF16; fall back to FP32 when BF16 is unsupported.

    FP32 avoids requiring FP16 gradient scaling for full fine-tuning, at the
    cost of additional memory. Use the same GPU for all experiment groups.
    """
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float32
