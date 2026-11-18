from functools import partial
import toolz
import einx
from jaxtyping import Float
import torch


class RmsNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.gain = torch.nn.Parameter(
            data=torch.zeros((d_model,), device=device, dtype=dtype)
        )

    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq dim"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms: Float[torch.Tensor, "batch seq"] = toolz.pipe(
            x,
            partial(einx.dot, "batch seq [dim], batch seq [dim] -> batch seq", x),
            lambda t: t / self.gain.data.size(0) + self.eps,
            torch.sqrt,
        )

        result = toolz.pipe(
            rms,
            partial(einx.divide, "batch seq dim, batch seq -> batch seq dim", x),
            partial(einx.multiply, "dim, batch seq dim -> batch seq dim", self.gain),
        )

        return result.to(in_dtype)
