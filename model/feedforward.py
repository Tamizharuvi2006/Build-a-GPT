import torch.nn as nn

from model.swiglu import SwiGLU
from config.model_config import FFN_MULTIPLIER

class FeedForward(nn.Module):

    def __init__(self, dim):

        super().__init__()

        hidden = int(dim * FFN_MULTIPLIER)

        self.net = SwiGLU(
            dim=dim,
            hidden_dim=hidden
        )

    def forward(self, x):

        return self.net(x)