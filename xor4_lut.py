from pathlib import Path

import numpy as np
from typing import Any, Dict, Tuple
from engine_context import EngineContext
from lut import load_coeff2d
from state_encoder import StateEncoder


class XOR4LUT:
    """
    4-bit XOR via bivariate polynomial LUT
    """
    def __init__(self, ctx: EngineContext, coeffs: np.ndarray):
        self.ctx = ctx
        self.sc = ctx.engine.slot_count
        self.coeffs = coeffs
        # cache plaintext coeffs
        self.pt = {}
        for p in range(16):
            for q in range(16):
                c = coeffs[p, q]
                if abs(c) > 1e-12:
                    vec = np.full(self.sc, c, dtype=np.complex128)
                    self.pt[(p, q)] = ctx.encode(vec)

    def _build_power_basis_16(self, ct: Any) -> Dict[int, Any]:
        """
        Build ct^k for k = 0..15 in Zeta16 domain using power basis + conjugate
        """
        eng = self.ctx
        # 1) make_power_basis 시도
        try:
            pos = eng.make_power_basis(ct, 8)
        except RuntimeError:
            # coeff(=INTT) 도메인 보장
            try:
                ct = eng.to_intt(ct)
            except RuntimeError:
                pass
            # 2) 재시도
            try:
                pos = eng.make_power_basis(ct, 8)
            except RuntimeError:
                # 3) 아직 부족 → coeff에서 bootstrap
                try:
                    ct = eng.to_intt(ct)
                except RuntimeError:
                    pass
                ct = eng.bootstrap(ct)
                pos = eng.make_power_basis(ct, 8)

        # ★ 곱셈 대신 뺄셈으로 0 만들기 (레벨 요구 X)
        zero_like = eng.sub(ct, ct)
        basis = {0: eng.add_plain(zero_like, 1.0)}
        for k in range(1, 9):
            basis[k] = pos[k - 1]
        for k in range(9, 16):
            basis[k] = eng.conjugate(pos[(16 - k) - 1])
        return basis


    def apply(self, a_ct, b_ct):
        eng = self.ctx
        A = self._build_power_basis_16(a_ct)
        B = self._build_power_basis_16(b_ct)

        # ★ 여기서도 곱셈 금지: A[0]과 같은 “안전한 레벨”에서 0 만들기
        res = eng.sub(A[0], A[0])

        for (p, q), pt in self.pt.items():
            term = eng.multiply(A[p], B[q])        # ct×ct (레벨 필요하지만 A,B는 안전화됨)
            res = eng.add(res, eng.multiply(term, pt))  # ct×pt (레벨 필요하지만 term이 충분)
        return res

    def __call__(self, a_ct, b_ct):
        return self.apply(a_ct, b_ct)


