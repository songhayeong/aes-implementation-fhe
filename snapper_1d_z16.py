from __future__ import annotations
from typing import Any, Dict
import json, numpy as np
from pathlib import Path
from engine_context import EngineContext


def load_coeff1d(json_path: Path) -> np.ndarray:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    max_k = max(int(k) for k, _, _ in obj["entries"])
    coeff = np.zeros(max_k + 1, dtype=np.complex128)
    for k, re, im in obj["entries"]:
        coeff[int(k)] = complex(re, im)
    return coeff


class Zeta16Snap1D:
    """
    Zeta 1D snap LUT : ct -> Poly(ct) 평가로 Zeta16 격자로 투영 (정규화)
    - 입력 : 단일 니블 (ciphertext)
    - 출력 : 정규화된 니블 (ciphertext)
    """
    def __init__(self, ctx: EngineContext, coeff_1d: np.ndarray, bootstrap_before: bool = False):
        self.ctx = ctx
        self.sc = ctx.engine.slot_count
        self.coeff = coeff_1d
        self.K = len(coeff_1d) - 1
        self.bootstrap_before = bootstrap_before

        # plaintext coeff cache
        self.pt: Dict[int, Any] = {}
        for k, c in enumerate(self.coeff):
            if abs(c) > 1e-12:
                vec = np.full(self.sc, c, dtype=np.complex128)
                self.pt[k] = ctx.encode(vec)

    def _power_basis_16(self, ct: Any) -> Dict[int, Any]:
        eng = self.ctx
        try:
            pos = eng.make_power_basis(ct, 8)  # returns [ct^1, ..., ct^8]
        except RuntimeError:
            ct = eng.bootstrap(ct)
            pos = eng.make_power_basis(ct, 8)

        # 0차항 만들기
        zero_like = eng.multiply(ct, 0.0)  # 실패 시 아래에서 bootstrap
        try:
            basis0 = eng.add_plain(zero_like, 1.0)
        except RuntimeError:
            ct = eng.bootstrap(ct)
            zero_like = eng.multiply(ct, 0.0)
            basis0 = eng.add_plain(zero_like, 1.0)

        basis = {0: basis0}
        for k in range(1, 9):
            basis[k] = pos[k - 1]
        for k in range(9, 16):
            basis[k] = eng.conjugate(pos[(16 - k) - 1])
        return basis

    def apply(self, ct: Any) -> Any:
        eng = self.ctx
        if self.bootstrap_before:
            ct = eng.bootstrap(ct)

        basis = self._power_basis_16(ct)

        # 결과 초기화
        try:
            res = eng.multiply(ct, 0.0)
        except RuntimeError:
            ct = eng.bootstrap(ct)
            basis = self._power_basis_16(ct)
            res = eng.multiply(ct, 0.0)

        # Σ_k c_k * (X^k)
        for k, pt in self.pt.items():
            term = basis[k] if k <= 15 else None
            if term is None:
                # 방어적 처리: k>15는 16으로 mod 축약 (X^16 ≈ 1) 가정
                term = basis[k % 16]
            res = eng.add(res, eng.multiply(term, pt))
        return res

class Zeta16SnapPair:
    """hi/lo 한쌍에 동일 1D 스냅 적용"""
    def __init__(self, snap1d: Zeta16Snap1D):
        self.snap = snap1d

    def apply_pair(self, ct_hi: Any, ct_lo: Any):
        return self.snap.apply(ct_hi), self.snap.apply(ct_lo)