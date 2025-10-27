"""Mask schedule functions R."""
import math

import numpy as np
import torch


def schedule(ratio, total_unknown, method="cosine"):
    """Generates a mask rate by scheduling mask functions R.

    Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. During
    training, the input ratio is uniformly sampled; during inference, the input
    ratio is based on the step number divided by the total iteration number: t/T.
    Based on experiements, we find that masking more in training helps.
    Args:
    ratio: The uniformly sampled ratio [0, 1) as input.
    total_unknown: The total number of tokens that can be masked out. For
        example, in MaskGIT, total_unknown = 256 for 256x256 images and 1024 for
        512x512 images.
    method: implemented functions are ["uniform", "cosine", "pow", "log", "exp"]
        "pow2.5" represents x^2.5

    Returns:
    The mask rate (float).
    """
    # total_unknown = torch.tensor(total_unknown, dtype=ratio.dtype, device=ratio.device)
    if method == "uniform":
        mask_ratio = 1. - ratio
    elif "linear" in method:
        mask_ratio = ratio
    elif "pow" in method:
        exponent = float(method.replace("pow", ""))
        mask_ratio = 1. - ratio**exponent
    elif method == "cosine":
        mask_ratio = torch.cos(math.pi / 2. * ratio)
    elif method == "log":
        mask_ratio = -torch.log2(ratio) / torch.log2(total_unknown)
    elif method == "exp":
        log2_total_unknown = torch.log2(total_unknown)
        mask_ratio = 1 - torch.exp2(-log2_total_unknown * (1 - ratio))
    # Clamps mask into [epsilon, 1)
    mask_ratio = torch.clamp(mask_ratio, 1e-6, 1.)
    return mask_ratio
