# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from models.codec.amphion_codec.quantize.factorized_vector_quantize import (
    FactorizedVectorQuantize,
)
class ResidualVQ(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "fvq",
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_type = quantizer_type

        if quantizer_type != "fvq":
            raise ValueError(f"Unsupported quantizer type: {quantizer_type}")

        self.quantizers = nn.ModuleList(
            [
                FactorizedVectorQuantize(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z):
        """Quantize residuals and return the quantized representation and indices."""
        quantized_out = 0.0
        residual = z
        all_indices = []
        for quantizer in self.quantizers:
            z_q, indices = quantizer(residual)
            quantized_out = quantized_out + z_q
            residual = residual - z_q
            all_indices.append(indices)
        return quantized_out, torch.stack(all_indices)
