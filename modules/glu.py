from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class GLU(nn.Module):
    r"""
    Gated Linear Unit. Provides a continuous spectrum of control rather than binary
    on/off (that a typical ReLU does)

    Calculated as:
    ```
    h = σ(W_gate · x) ⊙ (W_up · x)
    output = W_down · h
    ```
    where:
    - `⊙` = element-wise multiplication
    - `σ` = sigmoid function (outputs 0 to 1, the "gate")
    - `W_gate · x` = `gating_pathway` (just a variable holding the gating values,
    controls what gets through)
    - `W_up · x` = `signal_pathway` (upward projection, ups the dimension of x,
    carries the actual information)
    - `h` = hidden = `σ(gating_pathway) ⊙ signal_pathway`
    - `output` = projection back to original dimension with `W_out`
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        # mimicking how the pytorch code does this
        factory_kwargs = {"device": device, "dtype": dtype}

        self.W_gate = nn.Linear(
            in_features=hidden_dim, out_features=ffn_dim, bias=False, **factory_kwargs
        )
        self.W_up = nn.Linear(
            in_features=hidden_dim, out_features=ffn_dim, bias=False, **factory_kwargs
        )
        self.W_down = nn.Linear(
            in_features=ffn_dim, out_features=hidden_dim, bias=False, **factory_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        gating_pathway = self.W_gate(x)
        signal_pathway = self.W_up(x)
        hidden = torch.sigmoid(gating_pathway) * signal_pathway
        output = self.W_down(hidden)
        return output


class _GLU(nn.Module):
    r"""
    The same GLU, this time using PyTorch's own `torch.nn.functional.glu` for reference.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        # mimicking how the pytorch code does this
        factory_kwargs = {"device": device, "dtype": dtype}

        self.W_gate = nn.Linear(
            in_features=hidden_dim, out_features=ffn_dim, bias=False, **factory_kwargs
        )
        self.W_up = nn.Linear(
            in_features=hidden_dim, out_features=ffn_dim, bias=False, **factory_kwargs
        )
        self.W_down = nn.Linear(
            in_features=ffn_dim, out_features=hidden_dim, bias=False, **factory_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        gating_pathway = self.W_gate(x)
        signal_pathway = self.W_up(x)
        hidden = _GLU._concatenate_and_get_glu(gating_pathway, signal_pathway)
        output = self.W_down(hidden)
        return output

    @staticmethod
    def _concatenate_and_get_glu(gate: Tensor, signal: Tensor) -> Tensor:
        return nn.functional.glu(torch.cat([signal, gate], dim=-1))


if __name__ == "__main__":
    torch.manual_seed(40)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    hidden_dim = 8
    ffn_dim = 16
    dtype = torch.float16
    print(f"Using device: {device}\ndata type: {dtype}")
    x = torch.rand(3, 8, device=device, dtype=dtype)
    glu = GLU(hidden_dim, ffn_dim, device, dtype)
    x_glu = glu(x)
    print(f"""Out:
    {x_glu.shape}""")
