# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.custom_op import CustomOp
from vllm_spyre_next.custom_ops.utils import convert


def _gelu_exact(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="none")


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


_gelu_exact_compiled = torch.compile(_gelu_exact, dynamic=False)
_gelu_tanh_compiled  = torch.compile(_gelu_tanh,  dynamic=False)


@CustomOp.register("spyre_gelu")
class SpyreGELU(CustomOp):

    _TARGET_DEVICE = torch.device("spyre")
    _TARGET_DTYPE  = torch.float16

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(
                f"SpyreGELU: `approximate` must be 'none' or 'tanh', got {approximate!r}"
            )
        self.approximate = approximate
        self._compiled_op = (
            _gelu_exact_compiled if approximate == "none" else _gelu_tanh_compiled
        )

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)

    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_spyre(self, x: torch.Tensor) -> torch.Tensor:
        original_device = x.device
        original_dtype  = x.dtype
        x_spyre = convert(x, self._TARGET_DEVICE, self._TARGET_DTYPE)
        out_spyre = self._compiled_op(x_spyre)
        return convert(out_spyre, original_device, original_dtype)
