# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        use_l2_normlize=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.use_l2_normlize = use_l2_normlize

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(
                self.codebook_dim, self.input_dim, kernel_size=1
            )

        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, z):
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)
        z_q = z_e + (z_q - z_e).detach()
        return self.out_project(z_q), indices

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        b, d, t = latents.shape
        encodings = latents.transpose(1, 2).reshape(b * t, d)
        codebook = self.codebook.weight

        # L2 normalize encodings and codebook
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        # Compute euclidean distance between encodings and codebook,
        # if use_l2_normlize is True, the distance is equal to cosine distance
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(b, t)
        z_q = self.decode_code(indices)

        return z_q, indices


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
