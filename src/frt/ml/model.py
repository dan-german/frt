from __future__ import annotations

import torch
from torch import nn

from frt.config import FRETS_PER_STRING, NUM_STRINGS


class OnsetsAndFramesModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        conv_channels: int = 32,
        num_strings: int = NUM_STRINGS,
        frets_per_string: int = FRETS_PER_STRING,
    ):
        super().__init__()
        self.num_strings = num_strings
        self.frets_per_string = frets_per_string
        self.output_size = num_strings * frets_per_string

        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels // 2),
            nn.ReLU(),
            nn.Conv2d(conv_channels // 2, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )
        self.shared_projection = nn.LazyLinear(hidden_size)
        self.shared_activation = nn.ReLU()
        self.onset_gru = nn.GRU(
            hidden_size,
            hidden_size,
            batch_first=True,
        )
        self.onset_head = nn.Linear(hidden_size, self.output_size)
        self.frame_gru = nn.GRU(
            hidden_size + self.output_size,
            hidden_size,
            batch_first=True,
        )
        self.frame_head = nn.Linear(hidden_size, self.output_size)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.conv(features)
        shared = shared.permute(0, 2, 1, 3).flatten(start_dim=2)
        shared = self.shared_activation(self.shared_projection(shared))

        onset_state, _ = self.onset_gru(shared)
        onset_logits = self.onset_head(onset_state)

        onset_condition = torch.sigmoid(onset_logits)
        frame_input = torch.cat([shared, onset_condition], dim=-1)
        frame_state, _ = self.frame_gru(frame_input)
        frame_logits = self.frame_head(frame_state)

        output_shape = (
            features.shape[0],
            features.shape[2],
            self.num_strings,
            self.frets_per_string,
        )
        return {
            "frame_logits": frame_logits.reshape(output_shape),
            "onset_logits": onset_logits.reshape(output_shape),
        }
