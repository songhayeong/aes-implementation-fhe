# mixcolumns_enc.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import numpy as np

from engine_context import EngineContext
from state_encoder import StateEncoder
from xor4_lut import XOR4LUT
from lut import load_coeff1d, load_coeff2d


# -------------------- helpers: GF(2^8) tables → CKKS poly coeff --------------------

def _gf_mul_table(const: int) -> List[int]:
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B
            b >>= 1
        return res
    return [gf_mul(x, const) for x in range(256)]

def _ifft_coeff_from_table(tbl: List[int]) -> np.ndarray:
    z256 = np.exp(-2j * np.pi / 256)
    samples = np.array([z256 ** y for y in tbl], dtype=np.complex128)
    return np.fft.ifft(samples)  # len 256


class MixColumnsEnc:
    """
    MixColumns (encryption) 독립 모듈.
    입력/출력: 니블 도메인(ζ16) ct_hi/ct_lo  →  out_hi/out_lo (ζ16)
    내부:
      1) ζ16→ζ256 : B = lift(lo) * hi
      2) 8→8 LUT  : B2 = 2·B, B3 = 3·B   (power-basis 공유)
      3) 256→16   : (h1,l1)=split(B), (h2,l2)=split(B2), (h3,l3)=split(B3)
      4) 행별 마스크 + (소스행→타겟행) 정렬(회전) 후 XOR4로 합성
         - Row0: 2·row0 ⊕ 3·row1 ⊕ 1·row2 ⊕ 1·row3
         - Row1: 1·row0 ⊕ 2·row1 ⊕ 3·row2 ⊕ 1·row3
         - Row2: 1·row0 ⊕ 1·row1 ⊕ 2·row2 ⊕ 3·row3
         - Row3: 3·row0 ⊕ 1·row1 ⊕ 1·row2 ⊕ 2·row3
    """

    def __init__(self,
                 ctx: EngineContext,
                 split_hi_coeff: np.ndarray,   # 256→16 (hi)
                 split_lo_coeff: np.ndarray,   # 256→16 (lo)
                 xor4: XOR4LUT):
        self.ctx = ctx
        self.sc = ctx.engine.slot_count
        self.stride = self.sc // 16
        self.xor4 = xor4

        tol = 1e-12

        # --- 8→8 LUT coeffs for ×2, ×3 (encryption) ---
        c2 = _ifft_coeff_from_table(_gf_mul_table(2))
        c3 = _ifft_coeff_from_table(_gf_mul_table(3))
        self.ks2 = [k for k, v in enumerate(c2) if abs(v) > tol and k != 0]
        self.ks3 = [k for k, v in enumerate(c3) if abs(v) > tol and k != 0]
        self.deg256 = min(max([0] + self.ks2 + self.ks3), 128)
        self.pt_c2: Dict[int, Any] = {k: ctx.encode(np.full(self.sc, c2[k], dtype=np.complex128)) for k in self.ks2}
        self.pt_c3: Dict[int, Any] = {k: ctx.encode(np.full(self.sc, c3[k], dtype=np.complex128)) for k in self.ks3}
        self.c20 = complex(c2[0])
        self.c30 = complex(c3[0])

        # --- 16→256 lo-lift: ζ16^l → ζ256^l ---
        z256 = np.exp(-2j * np.pi / 256)
        lift = np.fft.ifft(np.array([z256 ** k for k in range(16)], dtype=np.complex128))
        self.ks_lift = [k for k, v in enumerate(lift) if abs(v) > tol and k != 0]
        self.deg16 = min(max([0] + self.ks_lift), 8)
        self.pt_lift = {k: ctx.encode(np.full(self.sc, lift[k], dtype=np.complex128)) for k in self.ks_lift}
        self.c0_lift = complex(lift[0])

        # --- 256→16 split(hi/lo) ---
        self.split_hi = split_hi_coeff
        self.split_lo = split_lo_coeff
        self.c0_hi = complex(split_hi_coeff[0]) if len(split_hi_coeff) > 0 else 0.0 + 0.0j
        self.c0_lo = complex(split_lo_coeff[0]) if len(split_lo_coeff) > 0 else 0.0 + 0.0j
        self.pt_shi: Dict[int, Any] = {k: ctx.encode(np.full(self.sc, split_hi_coeff[k], dtype=np.complex128))
                                       for k in range(1, len(split_hi_coeff)) if abs(split_hi_coeff[k]) > tol}
        self.pt_slo: Dict[int, Any] = {k: ctx.encode(np.full(self.sc, split_lo_coeff[k], dtype=np.complex128))
                                       for k in range(1, len(split_lo_coeff)) if abs(split_lo_coeff[k]) > tol}
        self.ks_shi = sorted(self.pt_shi.keys())
        self.ks_slo = sorted(self.pt_slo.keys())
        self.ks_union_nz = sorted(set(self.ks_shi) | set(self.ks_slo))

        # --- row masks (row r indices: r + 4*c) ---
        self.pt_row: List[Any] = []
        for r in range(4):
            m = np.zeros(self.sc, dtype=np.complex128)
            for c in range(4):
                m[(r + 4 * c) * self.stride] = 1.0
            self.pt_row.append(ctx.encode(m))

        # profiling state
        self._do_profile = False
        self._bs_calls = 0
        self._bs_total = 0.0

    # ---- bootstrap with profiling ----
    def _bs(self, ct):
        t0 = time.perf_counter()
        y = self.ctx.bootstrap(ct)
        self._bs_calls += 1
        self._bs_total += (time.perf_counter() - t0)
        return y

    # ---- apply two 8→8 polynomials with a shared power basis ----
    def _apply_two_polys_shared_basis(self, B: Any) -> Tuple[Any, Any]:
        eng = self.ctx
        if self.deg256 == 0:
            B2 = eng.add_plain(eng.multiply(B, 0.0), self.c20)
            B3 = eng.add_plain(eng.multiply(B, 0.0), self.c30)
            return B2, B3
        # ensure we can build the basis; bootstrap lazily
        try:
            pos = eng.make_power_basis(B, self.deg256)
        except RuntimeError:
            B = self._bs(B)
            pos = eng.make_power_basis(B, self.deg256)

        # evaluate ×2 and ×3 with the same basis
        def eval_poly(c0: complex, pt_map: Dict[int, Any]) -> Any:
            res = eng.add_plain(eng.multiply(B, 0.0), c0)
            for k, pt in pt_map.items():
                bk = pos[k - 1] if k <= self.deg256 else eng.conjugate(pos[256 - k - 1])
                res = eng.add(res, eng.multiply(bk, pt))
            return res

        B2 = eval_poly(self.c20, self.pt_c2)
        B3 = eval_poly(self.c30, self.pt_c3)
        return B2, B3

    # ---- 16→256 lift (lo) ----
    def _lift_lo(self, ct_lo: Any) -> Any:
        eng = self.ctx
        res = eng.add_plain(eng.multiply(ct_lo, 0.0), self.c0_lift)
        if self.deg16 > 0:
            try:
                pos16 = eng.make_power_basis(ct_lo, self.deg16)
            except RuntimeError:
                ct_lo = self._bs(ct_lo)
                pos16 = eng.make_power_basis(ct_lo, self.deg16)
            for k, pt in self.pt_lift.items():
                bk = pos16[k - 1] if k <= self.deg16 else eng.conjugate(pos16[16 - k - 1])
                res = eng.add(res, eng.multiply(bk, pt))
        return res

    # ---- 256→16 split (hi/lo) on a byte-domain ciphertext ----
    def _split_8to4_basis(self, ct_b: Any) -> Tuple[Any, Any]:
        eng = self.ctx
        res_hi = eng.add_plain(eng.multiply(ct_b, 0.0), self.c0_hi)
        res_lo = eng.add_plain(eng.multiply(ct_b, 0.0), self.c0_lo)
        if not self.ks_union_nz:
            return res_hi, res_lo

        max_k = max(self.ks_union_nz)
        deg = min(max_k, 128)
        try:
            pos = eng.make_power_basis(ct_b, deg)
        except RuntimeError:
            ct_b = self._bs(ct_b)
            pos = eng.make_power_basis(ct_b, deg)

        for k in self.ks_shi:
            bk = pos[k - 1] if k <= deg else eng.conjugate(pos[256 - k - 1])
            res_hi = eng.add(res_hi, eng.multiply(bk, self.pt_shi[k]))
        for k in self.ks_slo:
            bk = pos[k - 1] if k <= deg else eng.conjugate(pos[256 - k - 1])
            res_lo = eng.add(res_lo, eng.multiply(bk, self.pt_slo[k]))
        return res_hi, res_lo

    # ---- row-mask & alignment: source row s → target row t (같은 열 유지) ----
    def _align_row(self, ct: Any, s: int, t: int) -> Any:
        eng = self.ctx
        # (1) row s만 남기고
        part = eng.multiply(ct, self.pt_row[s])
        # (2) 같은 열에서 row s → row t 로 정렬: - (s - t) 칸 회전
        shift = (t - s) * self.stride
        return eng.rotate(part, shift)

    # ---- fold four terms via XOR4 with late bootstrap fallback ----
    def _xor4_fold4(self, a: Any, b: Any, c: Any, d: Any) -> Any:
        eng = self.ctx
        L = self.xor4(a, b)
        R = self.xor4(c, d)
        try:
            return self.xor4(L, R)
        except RuntimeError:
            return self.xor4(self._bs(L), self._bs(R))

    # ---- main ----
    def __call__(self, ct_hi: Any, ct_lo: Any, profile: bool = False) -> Tuple[Any, Any]:
        eng = self.ctx
        self._do_profile = profile
        self._bs_calls = 0
        self._bs_total = 0.0

        stamps = []
        def snap(lbl: str):
            if profile:
                stamps.append((lbl, time.perf_counter(), self._bs_calls, self._bs_total))

        # 1) ζ16→ζ256 재조합
        snap("start")
        lo_lift = self._lift_lo(ct_lo)
        B = eng.multiply(ct_hi, lo_lift)
        snap("recombine")

        # 2) *2,*3 (공유 basis)
        B2, B3 = self._apply_two_polys_shared_basis(B)
        snap("two_polys")

        # 3) split: a, 2a, 3a
        h1, l1 = self._split_8to4_basis(B)   # a
        h2, l2 = self._split_8to4_basis(B2)  # 2a
        h3, l3 = self._split_8to4_basis(B3)  # 3a
        snap("split3")

        # 4) 각 출력 행을 (소스행→타겟행) 정렬 후 XOR4로 합성
        # Row0 = [2,3,1,1] on src rows [0,1,2,3]
        r0_h = self._xor4_fold4(
            self._align_row(h2, 0, 0),
            self._align_row(h3, 1, 0),
            self._align_row(h1, 2, 0),
            self._align_row(h1, 3, 0),
        )
        r0_l = self._xor4_fold4(
            self._align_row(l2, 0, 0),
            self._align_row(l3, 1, 0),
            self._align_row(l1, 2, 0),
            self._align_row(l1, 3, 0),
        )

        # Row1 = [1,2,3,1]
        r1_h = self._xor4_fold4(
            self._align_row(h1, 0, 1),
            self._align_row(h2, 1, 1),
            self._align_row(h3, 2, 1),
            self._align_row(h1, 3, 1),
        )
        r1_l = self._xor4_fold4(
            self._align_row(l1, 0, 1),
            self._align_row(l2, 1, 1),
            self._align_row(l3, 2, 1),
            self._align_row(l1, 3, 1),
        )

        # Row2 = [1,1,2,3]
        r2_h = self._xor4_fold4(
            self._align_row(h1, 0, 2),
            self._align_row(h1, 1, 2),
            self._align_row(h2, 2, 2),
            self._align_row(h3, 3, 2),
        )
        r2_l = self._xor4_fold4(
            self._align_row(l1, 0, 2),
            self._align_row(l1, 1, 2),
            self._align_row(l2, 2, 2),
            self._align_row(l3, 3, 2),
        )

        # Row3 = [3,1,1,2]
        r3_h = self._xor4_fold4(
            self._align_row(h3, 0, 3),
            self._align_row(h1, 1, 3),
            self._align_row(h1, 2, 3),
            self._align_row(h2, 3, 3),
        )
        r3_l = self._xor4_fold4(
            self._align_row(l3, 0, 3),
            self._align_row(l1, 1, 3),
            self._align_row(l1, 2, 3),
            self._align_row(l2, 3, 3),
        )

        # 행별 블록은 슬롯이 겹치지 않으니 CKKS add로 합치면 최종
        out_hi = eng.add(eng.add(r0_h, r1_h), eng.add(r2_h, r3_h))
        out_lo = eng.add(eng.add(r0_l, r1_l), eng.add(r2_l, r3_l))
        snap("done")

        if profile and len(stamps) >= 2:
            print("\n[MixColumns profile]")
            for (lp, tp, cp, sp), (lc, tc, cc, sc) in zip(stamps, stamps[1:]):
                print(f" - {lp:10s} → {lc:10s}: wall {tc-tp:7.3f}s | BS +{cc-cp:2d}, +{sc-sp:7.3f}s")
            (l0, t0, c0, s0), (lN, tN, cN, sN) = stamps[0], stamps[-1]
            avg = (sN-s0)/(cN-c0) if (cN-c0) else 0.0
            print(f" - TOTAL      : wall {tN-t0:7.3f}s | BS +{cN-c0} calls, +{sN-s0:7.3f}s (avg {avg:5.3f}s)")
        return out_hi, out_lo


# -------------------- quick test harness --------------------

def _mix_single_col_plain(col: np.ndarray) -> List[int]:
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1:
                res ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi:
                a ^= 0x1B
            b >>= 1
        return res
    a0, a1, a2, a3 = map(int, col)
    return [
        gf_mul(a0, 2) ^ gf_mul(a1, 3) ^ a2 ^ a3,
        a0 ^ gf_mul(a1, 2) ^ gf_mul(a2, 3) ^ a3,
        a0 ^ a1 ^ gf_mul(a2, 2) ^ gf_mul(a3, 3),
        gf_mul(a0, 3) ^ a1 ^ a2 ^ gf_mul(a3, 2),
    ]

def _mixcolumns_plain(state16: np.ndarray) -> np.ndarray:
    out = []
    for c in range(4):
        col = state16[c*4:(c+1)*4]
        out.extend(_mix_single_col_plain(col))
    return np.array(out, dtype=np.uint8)


if __name__ == "__main__":
    # 1) 엔진/인코더/계수 로드
    ctx = EngineContext(signature=1, max_level=17, mode='cpu', thread_count=4)
    enc = StateEncoder(ctx)

    base = Path(__file__).parent / "generator" / "coeffs"
    split_hi = load_coeff1d(base / "split_mod256_to_16_hi.json")  # byte→hi4
    split_lo = load_coeff1d(base / "split_mod256_to_16_lo.json")  # byte→lo4
    xor4_coeffs = load_coeff2d(base / "xor4_coeffs.json", 16)
    xor4 = XOR4LUT(ctx, xor4_coeffs)

    mixc = MixColumnsEnc(ctx, split_hi, split_lo, xor4)

    # 2) 랜덤 상태 (column-first)
    np.random.seed(0)
    state = np.random.randint(0, 256, 16, dtype=np.uint8)

    # 데모용: SubBytes/ShiftRows 뒤라고 가정하고 니블 인코딩
    ct_hi, ct_lo = enc.encode(state)

    # 3) 실행 + 프로파일
    t0 = time.perf_counter()
    out_hi, out_lo = mixc(ct_hi, ct_lo, profile=True)
    t1 = time.perf_counter()
    dec = enc.decode(out_hi, out_lo)

    exp = _mixcolumns_plain(state)
    print("State        :", state)
    print("MixColumns   :", dec)
    print("Expected     :", exp)
    print("OK?          :", np.array_equal(dec, exp))
    print(f"Total time   : {t1 - t0:.3f}s")

    if not np.array_equal(dec, exp):
        bad = np.where(dec != exp)[0]
        print("Mismatch idx :", bad.tolist())
        print("pairs        :", [(dec[i], exp[i]) for i in bad])