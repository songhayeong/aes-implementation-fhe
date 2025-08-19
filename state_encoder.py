from typing import Tuple, Any

import numpy as np
from engine_context import EngineContext
from utils import ZetaEncoder


class StateEncoder:
    """
    Splits and packs 16-byte state into two ciphertexts (hi/lo nibble)
    """
    def __init__(self, ctx: EngineContext):
        self.ctx = ctx
        self.sc = ctx.engine.slot_count
        self.stride = self.sc // 16

    def encode(self, state: np.ndarray) -> Tuple[Any, Any]:
        assert state.shape == (16, )
        hi = (state >> 4) & 0x0F
        lo = state & 0x0F
        z_hi = ZetaEncoder.to_zeta(hi.astype(np.uint8), 16)
        z_lo = ZetaEncoder.to_zeta(lo.astype(np.uint8), 16)
        vec_hi = np.ones(self.sc, dtype=np.complex128)
        vec_lo = np.ones(self.sc, dtype=np.complex128)
        for i in range(16):
            vec_hi[i*self.stride] = z_hi[i]
            vec_lo[i*self.stride] = z_lo[i]
        return self.ctx.encrypt(vec_hi), self.ctx.encrypt(vec_lo)

    def decode(self, ct_hi, ct_lo) -> np.ndarray:
        vec_hi = self.ctx.decrypt(ct_hi)
        vec_lo = self.ctx.decrypt(ct_lo)
        out = np.empty(16, dtype=np.uint8)
        for i in range(16):
            hi_val = ZetaEncoder.from_zeta(np.array([vec_hi[i*self.stride]]), 16)[0]
            lo_val = ZetaEncoder.from_zeta(np.array([vec_lo[i*self.stride]]), 16)[0]
            out[i] = (hi_val << 4) | lo_val
        return out


