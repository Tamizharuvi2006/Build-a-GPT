import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):

    def __init__(self, dim, max_seq_len=4096):
        super().__init__()

        inv_freq = 1.0 / (
            10000 ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos", torch.cos(freqs))
        self.register_buffer("sin", torch.sin(freqs))

    def rotate_half(self, x):

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(self, x, offset=0):

        seq_len = x.shape[-2]

        cos = self.cos[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rot = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        )

        return x_rot.flatten(-2)