import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch

from model.llm import FantasyLLM

model = FantasyLLM()

x = torch.randint(
    0,
    8000,
    (2,128)
)

y = model(x)

print(y.shape)