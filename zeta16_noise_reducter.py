# zeta16_noise_reducer.py
from typing import Any, Dict
import numpy as np


class Zeta16NoiseReducer:
    """
    f(x) = -(1/16) x^17 + (17/16) x
    - ζ16 코드워드 근방에서 스냅/노이즈 축소
    - 내부에서 x^8까지 power basis를 만들고 x^16, x^17을 합성
    """

    def __init__(self, ctx, bootstrap_before: bool = False, bootstrap_after: bool = False):
        self.ctx = ctx
        self.alpha = 17.0 / 16.0  # (1 + 1/n)
        self.beta = -1.0 / 16.0  # -(1/n)
        self.bootstrap_before = bootstrap_before
        self.bootstrap_after = bootstrap_after

    def _ensure_power_basis(self, ct: Any):
        eng = self.ctx
        try:
            pos = eng.make_power_basis(ct, 8)  # x^1..x^8
            return ct, pos
        except RuntimeError:
            # 레벨/NTT 문제시 부트스트랩 후 재시도
            ct = eng.bootstrap(ct)
            pos = eng.make_power_basis(ct, 8)
            return ct, pos

    def apply(self, ct: Any) -> Any:
        eng = self.ctx
        x = ct

        if self.bootstrap_before:
            x = eng.bootstrap(x)

        # x^1..x^8
        x, pos = self._ensure_power_basis(x)
        x1 = pos[0]  # x
        x8 = pos[7]  # x^8

        # x^16 = (x^8)^2 , x^17 = x * x^16
        x16 = eng.multiply(x8, x8)
        x17 = eng.multiply(x16, x1)

        # f(x) = alpha * x  +  beta * x^17
        t1 = eng.multiply_plain(x1, self.alpha)
        t2 = eng.multiply_plain(x17, self.beta)
        y = eng.add(t1, t2)

        if self.bootstrap_after:
            y = eng.bootstrap(y)
        return y

    def apply_pair(self, ct_hi: Any, ct_lo: Any):
        return self.apply(ct_hi), self.apply(ct_lo)


# zeta16_snap_nomul.py
from typing import Any, Dict


class Zeta16SnapNoMul:
    """
    f(x) = (9/8) x + (1/8) x^9
    - 곱( ct×ct ) 없이 스냅/노이즈 축소
    - x^9는 power-basis의 x^7에 켤레(conjugate)로 얻음
    """

    def __init__(self, ctx, bootstrap_before=False, bootstrap_after=False):
        self.ctx = ctx
        self.a = 9.0 / 8.0
        self.b = 1.0 / 8.0
        self.bootstrap_before = bootstrap_before
        self.bootstrap_after = bootstrap_after

    def _pb1_8(self, ct: Any):
        eng = self.ctx
        try:
            return eng.make_power_basis(ct, 8)  # x^1..x^8
        except RuntimeError:
            ct = eng.bootstrap(ct)
            return eng.make_power_basis(ct, 8)

    def apply(self, ct: Any) -> Any:
        eng = self.ctx
        x = ct
        if self.bootstrap_before:
            x = eng.bootstrap(x)

        pos = self._pb1_8(x)  # pos[0]=x^1, ..., pos[7]=x^8
        x1 = pos[0]
        x9 = eng.conjugate(pos[6])

        y1 = eng.multiply_plain(x1, self.a)  # 스칼라곱
        y9 = eng.multiply_plain(x9, self.b)
        y = eng.add(y1, y9)

        if self.bootstrap_after:
            y = eng.bootstrap(y)
        return y

    def apply_pair(self, hi: Any, lo: Any):
        return self.apply(hi), self.apply(lo)


class Zeta16Snap:
    """
    f(x) = (17/16)x - (1/16)x^17  (ζ16 전체 고정점, f'(t)=0)
    x^17 = x^9 * x^8,  x^9 = conj(x^7)  → ct×ct 곱 1회
    """
    def __init__(self, ctx, *, always_bs: bool = False):
        self.ctx = ctx
        self.always_bs = always_bs  # True면 매번 한 번 부트스트랩하고 진행(안정/느림)

    def _to_coeff(self, ct: Any) -> Any:
        # 부트스트랩은 NTT가 아니어야 하므로 coeff 도메인 보장
        try:
            return self.ctx.to_intt(ct)
        except Exception:
            return ct

    def _pb_1_8(self, ct: Any):
        eng = self.ctx
        ct = self._to_coeff(ct)
        if self.always_bs:
            ct = eng.bootstrap(ct)
        try:
            pos = eng.make_power_basis(ct, 8)  # returns [x^1..x^8]
        except RuntimeError:
            ct = eng.bootstrap(ct)
            pos = eng.make_power_basis(ct, 8)
        return ct, pos

    def _mul_safe(self, a: Any, b: Any) -> Any:
        eng = self.ctx
        try:
            return eng.multiply(a, b)
        except RuntimeError:
            a = eng.bootstrap(a)
            return eng.multiply(a, b)

    def _scale_safe(self, ct: Any, s: float) -> Any:
        eng = self.ctx
        try:
            return eng.multiply(ct, float(s))
        except RuntimeError:
            ct = eng.bootstrap(ct)
            return eng.multiply(ct, float(s))

    def apply(self, ct: Any) -> Any:
        eng = self.ctx
        ct, pos = self._pb_1_8(ct)
        x1 = pos[0]        # x
        x8 = pos[7]        # x^8
        x9 = eng.conjugate(pos[6])

        x17 = self._mul_safe(x9, x8)
        t1  = self._scale_safe(x1, 17.0/16.0)
        t2  = self._scale_safe(x17, 1.0/16.0)

        try:
            return eng.sub(t1, t2)
        except AttributeError:
            return eng.add(t1, eng.multiply(t2, -1.0))

    def apply_pair(self, hi: Any, lo: Any):
        return self.apply(hi), self.apply(lo)