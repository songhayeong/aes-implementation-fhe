# shift rows
from typing import Any, List
import numpy as np
from engine_context import EngineContext


class ShiftRows:
    """
    AES ShiftRows (encryption)
     row 0 : shift 0
     row 1 : shift 1
     row 2 : shift 2
     row 3 : shift 3

     Packing 가정 : column-first (GHS12)
     linear indices: {a00, a10, a20, ... a13, a23, a33}
     -> row r indices : r -> 4 * c, c=0..3
    """

    def __init__(self, ctx: EngineContext):
        self.ctx = ctx
        self.sc = ctx.engine.slot_count
        self.stride = self.sc // 16  # gap packing stride

        # row masks (plaintext vectors) - 미리 encode 해서 캐시
        # mask[r][i*stride] = i if i in {r+4*c}
        self._pt_masks: List[Any] = []
        for r in range(4):
            mask = np.zeros(self.sc, dtype=np.complex128)
            for c in range(4):
                idx = (r + 4 * c) * self.stride
                mask[idx] = 1.0
            self._pt_masks.append(ctx.encode(mask))

        # 각 row의 좌회전 양 (column-major): s = r
        # 전체 벡터 회전 스텝 = -s * 4 * stride (왼쪽으로 s칸)
        self._rot_steps = [-r * 4 * self.stride for r in range(4)]

    def _apply_one(self, ct: Any) -> Any:
        eng = self.ctx
        # row별로: (ct * mask_r) 를 적절히 회전하고 모두 합산
        out = eng.multiply(ct, 0.0)
        for r in range(4):
            masked = eng.multiply(ct, self._pt_masks[r])  # ciphertext x plaintext(mask)
            if self._rot_steps[r] != 0:
                part = eng.rotate(masked, self._rot_steps[r])
            else:
                part = masked
            out = eng.add(out, part)
        return out

    def apply(self, ct_hi: Any, ct_lo: Any):
        """
        ct_hi / ct_lo 각각에 같은 Permutation 적용
        """
        return self._apply_one(ct_hi), self._apply_one(ct_lo)


# -------------------- for test -------------------------------
import numpy as np
from pathlib import Path
from engine_context import EngineContext
from state_encoder import StateEncoder
import time


def shiftrows_plain_colfirst(state16: np.ndarray) -> np.ndarray:
    # state16: [a00,a10,a20,a30, a01,a11,a21,a31, a02,a12,a22,a32, a03,a13,a23,a33]
    M = np.asarray(state16, dtype=np.uint8).reshape(4, 4, order='F')  # column-first → 4x4
    for r in range(4):
        M[r, :] = np.roll(M[r, :], -r)  # row r를 왼쪽으로 r만큼
    return M.reshape(16, order='F')  # 다시 column-first로 평탄화


def show_matrix_colfirst(state16):
    M = np.asarray(state16, dtype=np.uint8).reshape(4, 4, order='F')
    print(M)


def main():
    ctx = EngineContext(signature=2, max_level=17, mode='cpu', thread_count=4)
    enc = StateEncoder(ctx)
    shf = ShiftRows(ctx)

    # random state column-first
    np.random.seed(42)
    state = np.random.randint(0, 256, 16, dtype=np.uint8)
    print("State :", state)

    print("Before:")
    show_matrix_colfirst(state)

    ct_hi, ct_lo = enc.encode(state)
    t0 = time.perf_counter()
    out_hi, out_lo = shf.apply(ct_hi, ct_lo)
    t1 = time.perf_counter()
    dec = enc.decode(out_hi, out_lo)

    exp = shiftrows_plain_colfirst(state)

    print("After:")
    show_matrix_colfirst(dec)  # FHE ShiftRows 결과

    print("Expected:")
    show_matrix_colfirst(exp)  # 평문 ShiftRows 결과

    print("ShiftRows :", dec)
    print("Expected :", exp)
    print("Match? :", np.array_equal(dec, exp))
    print(f"ShiftRows took {t1 - t0:.6f} s")


if __name__ == "__main__":
    main()
