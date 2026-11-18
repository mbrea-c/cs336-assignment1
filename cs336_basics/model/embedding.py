import einx
from jaxtyping import Float, Int
import torch


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weights = torch.nn.Parameter(
            data=torch.nn.init.trunc_normal_(
                tensor=torch.zeros(
                    size=(num_embeddings, embedding_dim),
                    device=device,
                    dtype=dtype,
                ),
                mean=0,
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(  # ty: ignore[invalid-mutable-override]
        self, x: Int[torch.Tensor, "batch seq"]
    ) -> Float[torch.Tensor, "batch seq dim"]:
        return einx.get_at("[num] dim, batch seq -> batch seq dim", self.weights, x)
