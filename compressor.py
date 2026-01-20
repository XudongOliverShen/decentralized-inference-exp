from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import torch

@dataclass
class Payload:
    data: Any
    meta: Dict[str, Any]
    nbytes: int  # simulated traffic bytes

class Compressor:
    name = "base"
    def compress(self, x: torch.Tensor) -> Payload:
        raise NotImplementedError
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError

class NoneCompressor(Compressor):
    name = "none"
    def compress(self, x: torch.Tensor) -> Payload:
        return Payload(x, {}, x.numel() * x.element_size())
    def decompress(self, p: Payload, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return p.data.to(device=device)