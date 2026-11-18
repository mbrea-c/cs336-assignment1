import einx
import math
from jaxtyping import Float
from typing import override
import torch
import numpy as np


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        var = 2 / (in_features + out_features)
        std = math.sqrt(var)
        self.weights = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(
                tensor=torch.zeros(
                    size=(out_features, in_features),
                    device=device,
                    dtype=dtype,
                ),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    @override
    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: Float[torch.Tensor, "batch seq d_in"]
    ) -> Float[torch.Tensor, "batch seq d_out"]:
        # return einsum(x, self.weights, "batch seq d_in, d_out d_in -> batch seq d_out")
        return einx.dot(
            "batch seq [d_in], d_out [d_in] -> batch seq d_out", x, self.weights
        )
