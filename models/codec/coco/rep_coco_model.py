# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.nn as nn


from models.codec.amphion_codec.quantize import ResidualVQ
from models.codec.amphion_codec.vocos import VocosBackbone


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class CocoContentStyle(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        num_quantizers=1,
        quantizer_type="fvq",
        use_whisper=True,
        use_chromagram=True,
        cfg=None,
    ):
        super().__init__()

        assert cfg is not None
        self.cfg = cfg

        codebook_size = getattr(cfg, "codebook_size", codebook_size)
        hidden_size = getattr(cfg, "hidden_size", hidden_size)
        codebook_dim = getattr(cfg, "codebook_dim", codebook_dim)
        num_quantizers = getattr(cfg, "num_quantizers", num_quantizers)
        quantizer_type = getattr(cfg, "quantizer_type", quantizer_type)

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.num_quantizers = num_quantizers
        self.quantizer_type = quantizer_type

        if use_whisper:
            self.whisper_input_layer = nn.Linear(self.cfg.whisper_dim, hidden_size)
        if use_chromagram:
            self.chromagram_input_layer = nn.Linear(
                self.cfg.chromagram_dim, hidden_size
            )

        downsample_rate = getattr(cfg, "downsample_rate", 1)
        if downsample_rate > 1:
            self.do_downsample = True
            assert np.log2(downsample_rate).is_integer()

            down_layers = []
            up_layers = []
            for _ in range(int(np.log2(downsample_rate))):
                down_layers.extend(
                    [
                        nn.Conv1d(
                            hidden_size,
                            hidden_size,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.GELU(),
                    ]
                )
                up_layers.extend(
                    [
                        nn.ConvTranspose1d(
                            hidden_size, hidden_size, kernel_size=4, stride=2, padding=1
                        ),
                        nn.GELU(),
                    ]
                )
            self.downsample_layers = nn.Sequential(*down_layers)
            # Retain published checkpoint parameter names; not used in inference.
            self.upsample_layers = nn.Sequential(*up_layers)

        else:
            self.do_downsample = False

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.cfg.encoder.vocos_dim,
                intermediate_dim=self.cfg.encoder.vocos_intermediate_dim,
                num_layers=self.cfg.encoder.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.cfg.encoder.vocos_dim, self.hidden_size),
        )

        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_type=quantizer_type,
            use_l2_normlize=True,
        )

        self.reset_parameters()

    def forward(self, whisper_feats, chromagram_feats):
        x = self.whisper_input_layer(whisper_feats) + self.chromagram_input_layer(chromagram_feats)
        return self._quantize_features(x)

    def _quantize_features(self, x):
        if self.do_downsample:
            x = self.downsample_layers(x.transpose(1, 2)).transpose(1, 2)
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        quantized_out, all_indices = self.quantizer(x)
        if all_indices.shape[0] == 1:
            all_indices = all_indices.squeeze(0)
        return all_indices, quantized_out.transpose(1, 2)

    def quantize(self, whisper_feats, chromagram_feats):
        return self.forward(whisper_feats, chromagram_feats)

    def reset_parameters(self):
        self.apply(init_weights)


class CocoStyle(CocoContentStyle):
    def __init__(self, cfg):
        super().__init__(cfg=cfg, use_whisper=False, use_chromagram=True)

    def forward(self, chromagram_feats):
        return self._quantize_features(self.chromagram_input_layer(chromagram_feats))

    def quantize(self, chromagram_feats):
        return self.forward(chromagram_feats)
