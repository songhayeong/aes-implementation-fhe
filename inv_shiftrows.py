from __future__ import annotations
from typing import Any, Dict, Tuple, List
import numpy as np

from engine_context import EngineContext
from state_encoder import StateEncoder


class InvShiftRows:
    """
    AES InvShiftRows (decryption)
     row 0 : shift 0 to the right
     row 1 : shift 1 to the right
     row 2 : shift 2 to the right
     row 3 : shift 3 to the right

     Packing: column-first
    """

    def __init__(self, ctx: EngineContext):
        self.ctx = ctx
        self.sc = ctx.engine.slot_count
        self.stride = self.sc // 16

        # row masks (same as forward)
        self._pt_masks = []
        for r in range(4):
            m = np.zeros(self.sc, dtype=np.complex128)
            for c in range(4):
                idx = (r + 4 * c) * self.stride
                m[idx] = 1.0
            self._pt_masks.append(ctx.encode(m))

        # NOTE : 부호만 역방향
        self._rot_step = [+r * 4 * self.stride for r in range(4)]

    def _apply_one(self, ct: Any) -> Any:
        eng = self.ctx
        out = eng.multiply(ct, 0.0)
        for r in range(4):
            masked = eng.multiply(ct, self._pt_masks[r])
            part = eng.rotate(masked, self._rot_step[r]) if self._rot_step[r] else masked
            out = eng.add(out, part)
        return out

    def apply(self, ct_hi, ct_lo):
        return self._apply_one(ct_hi), self._apply_one(ct_lo)


# ---------- 컬럼-퍼스트 기준의 평문 레퍼런스 ----------
def shiftrows_plain_colfirst(state16: np.ndarray) -> np.ndarray:
    """ShiftRows (left by row index) on column-first 1D state (len=16)."""
    out = state16.copy()
    for r in range(4):
        cols = [state16[r + 4 * c] for c in range(4)]
        cols = np.roll(cols, -r)  # 왼쪽 r칸
        for c in range(4):
            out[r + 4 * c] = cols[c]
    return out


def invshiftrows_plain_colfirst(state16: np.ndarray) -> np.ndarray:
    """InvShiftRows (right by row index) on column-first 1D state (len=16)."""
    out = state16.copy()
    for r in range(4):
        cols = [state16[r + 4 * c] for c in range(4)]
        cols = np.roll(cols, +r)  # 오른쪽 r칸
        for c in range(4):
            out[r + 4 * c] = cols[c]
    return out


# ---------- print util ----------
def _print_cmp(tag: str, got: np.ndarray, exp: np.ndarray):
    ok = np.array_equal(got, exp)
    print(f"[{tag:12s}] OK? {ok} | got={got.tolist()} | exp={exp.tolist()}")
    if not ok:
        bad = np.where(got != exp)[0]
        print("  mismatch idx:", bad.tolist())
        print("  pairs:", [(int(got[i]), int(exp[i])) for i in bad])


# ---------- 테스트 ----------
if __name__ == "__main__":
    # 엔진/인코더 준비
    ctx = EngineContext(signature=1, max_level=17, mode="cpu", thread_count=4)
    enc = StateEncoder(ctx)

    from shift_rows import ShiftRows  # <- 파일/모듈명에 맞춰 수정

    sr = ShiftRows(ctx)
    isr = InvShiftRows(ctx)

    np.random.seed(0)
    state_cf = np.random.randint(0, 256, 16, dtype=np.uint8)

    ct_hi, ct_lo = enc.encode(state_cf)

    ct_hi_s, ct_lo_s = sr.apply(ct_hi, ct_lo)
    dec_s = enc.decode(ct_hi_s, ct_lo_s)
    exp_s = shiftrows_plain_colfirst(state_cf)
    _print_cmp("ShiftRows", dec_s, exp_s)

    ct_hi_is, ct_lo_is = isr.apply(ct_hi_s, ct_lo_s)
    dec_is = enc.decode(ct_hi_is, ct_lo_is)
    _print_cmp("Inv∘Shift==Id", dec_is, state_cf)
    exp_is = invshiftrows_plain_colfirst(exp_s)
    _print_cmp("InvShiftRows", dec_is, exp_is)

