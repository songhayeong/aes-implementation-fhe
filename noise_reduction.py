from __future__ import annotations
from typing import Any, Optional
import time

"""
    essential polynomial :
    f(x) = (1 + 1/16)x - (1/16)x^17
    
    implementation : x^16을 빠르게 만든 후, x^17 만들기
    -> 계수 스칼라곱 2번 + 더하기 1번
"""


class NoiseReducer:
    """
    Noise reduction f(x) = (1 + 1/n) x - (1/n) x^(n+1)
    - n = 16 (nibble domain)
    - level/NTT 에러 시 1회 Bootstrap 후 재시도
    - make_power_basis 가능하면 재사용, 불가하면 square chain
    """

    def __init__(self, ctx, n: int = 16, profile: bool = False):
        assert n >= 2
        self.ctx = ctx
        self.n = n
        self.alpha = 1.0 + 1.0 / n  # (1+1/n)
        self.beta = -1.0 / n  # (=1/n)
        self.profile = profile
        self._last_stats = None

    def _ensure_read(self, ct: Any, deg: int = 1) -> Any:
        """
        make_power_basis 가 가능한 상태인지 확인, 아니면 1회 bootstrap.
        """
        try:
            self.ctx.make_power_basis(ct, deg)
            return ct
        except RuntimeError:
            return self.ctx.bootstrap(ct)

    def _x_pow_nplus1(self, x: Any) -> Any:
        eng = self.ctx
        # 1) basis try
        if self.n == 16:
            try:
                pos = eng.make_power_basis(x, 16)
            except RuntimeError:
                x = eng.bootstrap(x)
                pos = eng.make_power_basis(x, 16)
            x16 = pos[15]
            return eng.relinearize(eng.multiply(x16, x))

    def apply(self, ct: Any) -> Any:
        """
        단일 암호문에 Noise reduction 적용
        """
        eng = self.ctx
        t0 = time.perf_counter() if self.profile else None

        # prepare
        x = self._ensure_read(ct, 1)

        # x^(n+1)
        xn1 = self._x_pow_nplus1(x)

        # f(x) = alpha * x + beta * xn1
        a = eng.multiply_plain(x, self.alpha)
        b = eng.multiply_plain(xn1, self.beta)
        y = eng.add(a, b)

        if self.profile:
            self._last_stats = {"wall_s": time.perf_counter() - t0}
        return y

    def apply_pair(self, ct_hi: Any, ct_lo: Any) -> tuple[Any, Any]:
        """
        nibble (hi, lo) 한 쌍에 NR 적용
        """
        return self.apply(ct_hi), self.apply(ct_lo)

    def last_profile(self) -> Optional[dict]:
        return self._last_stats
