#!/usr/bin/env python3
#  2021, Carnegie Mellon University;  Xuankai Chang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Linear Projection."""

from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling


class Subsampling(AbsPreEncoder):
    """Conv Preencoder w/ subsampling."""

    def __init__(self, input_size: int, output_size: int, dropout: float = 0.0):
        """Initialize the module."""
        assert check_argument_types()
        super().__init__()

        self.subsampling = Conv2dSubsampling(
            input_size, output_size, dropout, torch.nn.Identity()
        )
        self.output_dim = output_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward."""
        output, _ = self.subsampling(input, x_mask=None)
        return output, ((input_lengths - 1) // 2 - 1) // 2  # input_lengths - 2

    def output_size(self) -> int:
        """Get the output size."""
        return self.output_dim
