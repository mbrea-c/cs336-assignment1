import torch
from einops import rearrange, einsum


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ):
        # TODO: I forgot how to do OOP in Python lol. How to call constructor of
        # parent class?
        torch.nn.Module.__init__(self, in_features, out_features, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
