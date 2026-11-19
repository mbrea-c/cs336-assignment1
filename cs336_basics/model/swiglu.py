import einx
from jaxtyping import Float
import torch


class SwiGlu(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        d_ff = max(1, round(3 / 8 * d_model / 64)) * 64 if d_ff is None else d_ff
        self.w1 = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(
                tensor=torch.zeros((d_ff, d_model), device=device, dtype=dtype)
            )
        )
        self.w3 = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(
                tensor=torch.zeros((d_ff, d_model), device=device, dtype=dtype)
            )
        )
        self.w2 = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(
                tensor=torch.zeros((d_model, d_ff), device=device, dtype=dtype)
            )
        )

    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq dim"]:
        silu_linear = einx.dot(
            "dff [dim], batch seq [dim] -> batch seq dff", self.w1, x
        )
        silu = torch.sigmoid(silu_linear) * silu_linear
        gate = einx.dot("dff [dim], batch seq [dim] -> batch seq dff", self.w3, x)
        return einx.dot(
            "dim [dff], batch seq [dff] -> batch seq dim", self.w2, silu * gate
        )
