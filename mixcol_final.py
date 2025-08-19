from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple, List
import json
import numpy as np

from engine_context import EngineContext
from state_encoder import StateEncoder  # 기존 니블 인/디코더
from xor4_lut import XOR4LUT
from zeta16_noise_reducter import Zeta16NoiseReducer, Zeta16SnapNoMul, Zeta16Snap


COEFF_DIR = Path(__file__).parent / "generator" / "coeffs"


# --------- 2-variable coeff cache (gf×k용) ----------
class _CoeffCache:
    def __init__(self):
        self.pt_cache: Dict[Tuple[int, str], Dict[Tuple[int, int], Any]] = {}

    def load_plaintexts(self, ctx: EngineContext, mult: int, which: str) -> Dict[Tuple[int, int], Any]:
        key = (mult, which)
        if key in self.pt_cache:
            return self.pt_cache[key]
        path = COEFF_DIR / f"gf_mult{mult}_{which}_coeffs.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        entries = obj["entries"]

        sc = ctx.engine.slot_count
        d: Dict[Tuple[int, int], Any] = {}
        for p, q, re, im in entries:
            c = complex(re, im)
            d[(p, q)] = ctx.encode(np.full(sc, c, dtype=np.complex128))
        self.pt_cache[key] = d
        return d


# ---------------- MixColumns (XOR 주입형) ----------------
class MixColFinal:
    """
    입력/출력: ζ16 니블 (ct_hi, ct_lo) -> (out_hi, out_lo)
    내부: (ct_hi, ct_lo)에 대해 2변수 폴리로 gf×2(원본), gf×3(열-위로1칸) 평가,
         그 뒤 row-major 상태에서 column-shift(열 위로 2,3칸)와 XOR 누적.
    XOR은 AddRoundKey에서 쓰는 XOR4LUT.apply를 그대로 주입받아 사용.
    """

    def __init__(self, ctx: EngineContext, xor4: XOR4LUT, stride: int | None = None):
        self.ctx = ctx
        self.xor4 = xor4  # ★ AddRoundKey와 동일 구현을 그대로 사용
        self.sc = ctx.engine.slot_count
        self.stride = stride if stride is not None else (self.sc // 16)
        self._coeffs = _CoeffCache()
        self.enc = StateEncoder(ctx)
        zeros = np.zeros(16, dtype=np.uint8)
        self.zero_hi, self.zero_lo = self.enc.encode(zeros)

    def _normalize_via_xor_zero(self, ct, which: str):
        zero_ct = self.zero_hi if which == 'hi' else self.zero_lo
        return self._xor_ct(ct, zero_ct)

    # ζ16 power-basis (0..15). 1..8은 power, 9..15는 켤레
    def _basis16(self, ct: Any) -> Dict[int, Any]:
        eng = self.ctx
        try:
            pos = eng.make_power_basis(ct, 8)
        except RuntimeError:
            ct = eng.bootstrap(ct)
            pos = eng.make_power_basis(ct, 8)
        zero_like = eng.multiply(ct, 0.0)
        basis: Dict[int, Any] = {0: eng.add_plain(zero_like, 1.0)}
        for k in range(1, 9):
            basis[k] = pos[k - 1]
        for k in range(9, 16):
            basis[k] = eng.conjugate(pos[(16 - k) - 1])
        return basis

    # Σ c[p,q] * X^p * Y^q  (gf×k 평가)
    def _gf_poly_eval_2var(self, ct_hi: Any, ct_lo: Any, mult: int, which: str) -> Any:
        eng = self.ctx
        basis_x = self._basis16(ct_hi)
        basis_y = self._basis16(ct_lo)
        coeff_pt = self._coeffs.load_plaintexts(eng, mult, which)

        res = eng.multiply(ct_hi, 0.0)
        for (p, q), pt in coeff_pt.items():
            term = eng.multiply(basis_x[p], basis_y[q])
            term = eng.multiply(term, pt)
            res = eng.add(res, term)
        return res

    def gf_mult_2(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return (self._gf_poly_eval_2var(ct_hi, ct_lo, 2, "hi"),
                self._gf_poly_eval_2var(ct_hi, ct_lo, 2, "lo"))

    def gf_mult_3(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return (self._gf_poly_eval_2var(ct_hi, ct_lo, 3, "hi"),
                self._gf_poly_eval_2var(ct_hi, ct_lo, 3, "lo"))

    def _col_shift_rowmajor(self, ct: Any, k_up: int) -> Any:
        return self.ctx.rotate(ct, -4 * k_up * self.stride)

    def _renorm_pair(self, hi: Any, lo: Any) -> Tuple[Any, Any]:
        state = self.enc.decode(hi, lo)
        return self.enc.encode(state)

    def _xor_ct(self, a: Any, b: Any) -> Any:
        return self.xor4.apply(a, b)

    # ---- main ----
    def __call__(self, ct_hi: Any, ct_lo: Any,
                 do_final_bootstrap: bool = True,
                 debug: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        dbg = debug if isinstance(debug, dict) else None

        def log(k, v):
            if dbg is not None: dbg[k] = v

        # ct_hi = self.ctx.to_intt(ct_hi)
        # ct_lo = self.ctx.to_intt(ct_lo)

        # 1) 열 시프트 준비 (row-major 기준, 열 위로 1/2/3칸)
        r1_hi, r1_lo = self._col_shift_rowmajor(ct_hi, 1), self._col_shift_rowmajor(ct_lo, 1)
        r2_hi, r2_lo = self._col_shift_rowmajor(ct_hi, 2), self._col_shift_rowmajor(ct_lo, 2)
        r3_hi, r3_lo = self._col_shift_rowmajor(ct_hi, 3), self._col_shift_rowmajor(ct_lo, 3)
        log("rotc1", (r1_hi, r1_lo))
        log("rotc2", (r2_hi, r2_lo))
        log("rotc3", (r3_hi, r3_lo))
        log("in", (ct_hi, ct_lo))

        # 2) gf×2(원본), gf×3(열-위로1칸)
        two_hi, two_lo = self.gf_mult_2(ct_hi, ct_lo)
        thr_hi, thr_lo = self.gf_mult_3(r1_hi, r1_lo)
        log("two", (two_hi, two_lo));
        log("thr", (thr_hi, thr_lo))

        # 3) XOR 누적 (AddRoundKey XOR 그대로)
        acc1_hi = self._xor_ct(two_hi, thr_hi);
        acc1_lo = self._xor_ct(two_lo, thr_lo);
        log("acc1", (acc1_hi, acc1_lo))

        acc1_hi, acc1_lo = self._renorm_pair(acc1_hi, acc1_lo)
        # r2_hi, r2_lo = self._renorm_pair(r2_hi, r2_lo)
        # r3_hi, r3_lo = self._renorm_pair(r3_hi, r3_lo)

        acc2_hi = self._xor_ct(acc1_hi, r2_hi)
        acc2_lo = self._xor_ct(acc1_lo, r2_lo)
        log("acc2", (acc2_hi, acc2_lo))

        acc2_hi, acc2_lo = self._renorm_pair(acc2_hi, acc2_lo)
        acc3_hi = self._xor_ct(acc2_hi, r3_hi)
        acc3_lo = self._xor_ct(acc2_lo, r3_lo)
        acc3_hi, acc3_lo = self._renorm_pair(acc3_hi, acc3_lo)
        log("acc3", (acc3_hi, acc3_lo))
        out_hi, out_lo = acc3_hi, acc3_lo
        # 예시: 마지막에 안정성용
        if do_final_bootstrap:
            out_hi = self.ctx.to_intt(out_hi)
            out_hi = self.ctx.bootstrap(out_hi)
            out_lo = self.ctx.to_intt(out_lo)
            out_lo = self.ctx.bootstrap(out_lo)
            log("out", (out_hi, out_lo))

        return out_hi, out_lo


# -------------------- 평문 레퍼런스/검증 --------------------
def _gf_mul2_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint16)
    red = ((v & 0x80) != 0).astype(np.uint16) * 0x1B
    return (((v << 1) & 0xFF) ^ red).astype(np.uint8)


def _gf_mul3_vec(v: np.ndarray) -> np.ndarray:
    return (_gf_mul2_vec(v) ^ v).astype(np.uint8)


def _rotate_rows_in_col_plain(state16: np.ndarray, k: int) -> np.ndarray:
    out = state16.copy()
    for c in range(4):
        s = c  # row-major에서 열 c의 원소 인덱스: c, c+4, c+8, c+12
        col = out[[s, s + 4, s + 8, s + 12]]
        col = np.roll(col, -k)  # 위로 k칸 ⇒ -k
        out[[s, s + 4, s + 8, s + 12]] = col
    return out


def _mix_single_col_plain(col: np.ndarray) -> List[int]:
    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1: res ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi: a ^= 0x1B
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
    # row-major에서 한 열은 (c, c+4, c+8, c+12)
    for c in range(4):
        col = state16[[c, c + 4, c + 8, c + 12]]
        out.extend(_mix_single_col_plain(col))
    # 다시 row-major로 배치
    out = np.array(out, dtype=np.uint8)
    # out은 열별 결과가 순차로 붙은 상태 → row-major로 재배치
    rm = np.empty(16, dtype=np.uint8)
    for c in range(4):
        rm[[c, c + 4, c + 8, c + 12]] = out[c * 4:(c + 1) * 4]
    return rm


def _print_cmp(tag: str, got: np.ndarray, exp: np.ndarray):
    ok = np.array_equal(got, exp)
    print(f"[{tag:10s}] OK? {ok} | got={got.tolist()} | exp={exp.tolist()}")
    if not ok:
        bad = np.where(got != exp)[0]
        print(f"  mismatch idx: {bad.tolist()}")
        print("  pairs:", [(int(got[i]), int(exp[i])) for i in bad])


def dec_pair_safe(name: str, dbg: Dict[str, Any], enc: StateEncoder):
    if name not in dbg: return None
    try:
        h, l = dbg[name]
        return enc.decode(h, l)
    except Exception:
        return None


def print_cmp_safe(tag: str, got: np.ndarray | None, exp: np.ndarray):
    if got is None:
        print(f"[{tag:10s}] SKIP");
        return
    _print_cmp(tag, got, exp)


# -------------------- 테스트 --------------------
if __name__ == "__main__":
    ctx = EngineContext(signature=1, max_level=17, mode="cpu", thread_count=4)
    enc = StateEncoder(ctx)

    # XOR4 계수 로드
    xor4_coeffs = json.loads((COEFF_DIR / "xor4_coeffs.json").read_text(encoding="utf-8"))
    coeff_mat = np.zeros((16, 16), dtype=np.complex128)
    for p, q, re, im in xor4_coeffs["entries"]:
        coeff_mat[p, q] = complex(re, im)
    xor4 = XOR4LUT(ctx, coeff_mat)

    mixc = MixColFinal(ctx, xor4)

    # 입력
    np.random.seed(0)
    state = np.random.randint(0, 256, 16, dtype=np.uint8)
    ct_hi, ct_lo = enc.encode(state)

    # 평문 레퍼런스
    r1_p = _rotate_rows_in_col_plain(state, 1)  # 열 "위로" 1칸
    r2_p = _rotate_rows_in_col_plain(state, 2)
    r3_p = _rotate_rows_in_col_plain(state, 3)
    two_p = _gf_mul2_vec(state)
    thr_p = _gf_mul3_vec(r1_p)
    acc1_p = (two_p ^ thr_p).astype(np.uint8)
    acc2_p = (acc1_p ^ r2_p).astype(np.uint8)
    acc3_p = (acc2_p ^ r3_p).astype(np.uint8)
    mix_p = _mixcolumns_plain(state)

    # 실행 + 비교
    dbg: Dict[str, Any] = {}
    t0 = time.perf_counter()
    out_hi, out_lo = mixc(ct_hi, ct_lo, do_final_bootstrap=True, debug=dbg)
    t1 = time.perf_counter()

    _print_cmp("input", enc.decode(ct_hi, ct_lo), state)
    print_cmp_safe("rotc1", dec_pair_safe("rotc1", dbg, enc), r1_p)
    print_cmp_safe("rotc2", dec_pair_safe("rotc2", dbg, enc), r2_p)
    print_cmp_safe("rotc3", dec_pair_safe("rotc3", dbg, enc), r3_p)
    print_cmp_safe("gf*2", dec_pair_safe("two", dbg, enc), two_p)
    print_cmp_safe("gf*3@c1", dec_pair_safe("thr", dbg, enc), thr_p)
    print_cmp_safe("acc1", dec_pair_safe("acc1", dbg, enc), acc1_p)
    print_cmp_safe("acc2", dec_pair_safe("acc2", dbg, enc), acc2_p)
    print_cmp_safe("acc3", dec_pair_safe("acc3", dbg, enc), acc3_p)

    dec_out = enc.decode(out_hi, out_lo)
    _print_cmp("final", dec_out, mix_p)
    print(f"Total time: {t1 - t0:.3f}s")
