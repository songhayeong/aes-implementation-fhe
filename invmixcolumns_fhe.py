from __future__ import annotations

import time
from typing import Any, Dict, Tuple, List
import numpy as np
from pathlib import Path
import json

from engine_context import EngineContext
from state_encoder import StateEncoder
from xor4_lut import XOR4LUT
from mixcol_final import _print_cmp, dec_pair_safe, print_cmp_safe, _rotate_rows_in_col_plain

COEFF_DIR = Path(__file__).parent / "generator" / "coeffs"


class _CoeffCache:
    def __init__(self):
        self.pt_cache: Dict[Tuple[int, str], Dict[Tuple[int, int], Any]] = {}

    def load_plaintexts(self, ctx: EngineContext, mult: int, which: str) -> Dict[Tuple[int, int], Any]:
        key = (mult, which)
        if key in self.pt_cache: return self.pt_cache[key]
        path = COEFF_DIR / f"gf_mult{mult}_{which}_coeffs.json"
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        sc = ctx.engine.slot_count
        d: Dict[Tuple[int, int], Any] = {}
        for p, q, re, im in entries:
            d[(p, q)] = ctx.encode(np.full(sc, complex(re, im), dtype=np.complex128))
        self.pt_cache[key] = d
        return d


class InvMixColumnsFHE:
    """
    AES InvMixColumns (decryption) for column-first packed state (length 16).
    - Inputs: (ct_hi, ct_lo) in ζ16 nibble domain
    - Matrix: [[0E,0B,0D,09],[09,0E,0B,0D],[0D,09,0E,0B],[0B,0D,09,0E]]
    - 구현: r0=orig, r1=row-rotate(-1), r2=-2, r3=-3
            out = 14*r0 ⊕ 11*r1 ⊕ 13*r2 ⊕  9*r3
    """

    def __init__(self, ctx: EngineContext, xor4: XOR4LUT, use_hard_renorm: bool = True):
        self.ctx = ctx
        self.xor4 = xor4
        self.sc = ctx.engine.slot_count
        self.stride = self.sc // 16
        self._coeffs = _CoeffCache()
        self.enc = StateEncoder(ctx)
        self.use_hard_renorm = use_hard_renorm

        # row masks: row r indices = r + 4*c  (column-first)
        self._pt_row: List[Any] = []
        for r in range(4):
            m = np.zeros(self.sc, dtype=np.complex128)
            for c in range(4):
                idx = (r + 4 * c) * self.stride
                m[idx] = 1.0
            self._pt_row.append(ctx.encode(m))

    # ----- ζ16 power-basis (0..15) -----
    def _basis16(self, ct: Any) -> Dict[int, Any]:
        eng = self.ctx
        try:
            pos = eng.make_power_basis(ct, 8)
        except RuntimeError:
            ct = eng.bootstrap(ct);
            pos = eng.make_power_basis(ct, 8)
        zero = eng.multiply(ct, 0.0)
        basis = {0: eng.add_plain(zero, 1.0)}
        for k in range(1, 9): basis[k] = pos[k - 1]
        for k in range(9, 16): basis[k] = eng.conjugate(pos[(16 - k) - 1])
        return basis

    # ----- 2-var poly eval: Σ c[p,q] X^p Y^q -----
    def _poly2_eval(self, ct_hi: Any, ct_lo: Any, mult: int, which: str) -> Any:
        eng = self.ctx
        bx = self._basis16(ct_hi);
        by = self._basis16(ct_lo)
        coeff_pt = self._coeffs.load_plaintexts(eng, mult, which)
        acc = eng.multiply(ct_hi, 0.0)
        for (p, q), pt in coeff_pt.items():
            term = eng.multiply(bx[p], by[q])
            term = eng.multiply(term, pt)
            acc = eng.add(acc, term)
        return acc

    # ----- XOR helper -----
    def _xor(self, a: Any, b: Any) -> Any:
        # LUT XOR (4-bit) — 필요시 내부에서 레벨 보정
        return self.xor4.apply(a, b)

    # ----- 하드 리노멀(안정성): decode→re-encode (느리지만 튼튼) -----
    def _renorm_pair(self, hi: Any, lo: Any) -> Tuple[Any, Any]:
        if not self.use_hard_renorm: return hi, lo
        st = self.enc.decode(hi, lo)
        return self.enc.encode(st)

    # ----- row-rotate within each column: r -> r+k (column 고정) -----
    def _rot_rows_in_col(self, ct: Any, k_rows: int) -> Any:
        eng = self.ctx
        parts = []
        # row r만 남기고, 같은 column들에서 r→r+k : 인덱스 이동량 = k*stride
        for r in range(4):
            masked = eng.multiply(ct, self._pt_row[r])
            parts.append(eng.rotate(masked, k_rows * self.stride))
        out = eng.multiply(ct, 0.0)
        for p in parts: out = eng.add(out, p)
        return out

    def gf_mult_9(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return (self._poly2_eval(ct_hi, ct_lo, 9, "hi"),
                self._poly2_eval(ct_hi, ct_lo, 9, "lo"))

    def gf_mult_11(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return (self._poly2_eval(ct_hi, ct_lo, 11, "hi"),
                self._poly2_eval(ct_hi, ct_lo, 11, "lo"))

    def gf_mult_13(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return (self._poly2_eval(ct_hi, ct_lo, 13, "hi"),
                self._poly2_eval(ct_hi, ct_lo, 13, "lo"))

    def gf_mult_14(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return (self._poly2_eval(ct_hi, ct_lo, 14, "hi"),
                self._poly2_eval(ct_hi, ct_lo, 14, "lo"))

    # row-major에서 열 위로 k칸 shift
    def _col_shift_rowmajor(self, ct: Any, k_up: int) -> Any:
        return self.ctx.rotate(ct, -4 * k_up * self.stride)

    def __call__(self, ct_hi: Any, ct_lo: Any,
                 do_final_bootstrap: bool = True,
                 debug: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        eng = self.ctx

        def log(k, v):
            if debug is not None: debug[k] = v

        r0h, r0l = ct_hi, ct_lo
        r1h, r1l = self._col_shift_rowmajor(ct_hi, 1), self._col_shift_rowmajor(ct_lo, 1)
        r2h, r2l = self._col_shift_rowmajor(ct_hi, 2), self._col_shift_rowmajor(ct_lo, 2)
        r3h, r3l = self._col_shift_rowmajor(ct_hi, 3), self._col_shift_rowmajor(ct_lo, 3)
        log("rotc1", (r1h, r1l))
        log("rotc2", (r2h, r2l))
        log("rotc3", (r3h, r3l))

        e14h, e14l = self.gf_mult_14(r0h, r0l); log("mul14", (e14h, e14l))
        e11h, e11l = self.gf_mult_11(r1h, r1l); log("mul11", (e11h, e11l))
        e13h, e13l = self.gf_mult_13(r2h, r2l); log("mul13", (e13h, e13l))
        e9h, e9l = self.gf_mult_9(r3h, r3l); log("mul9", (e9h, e9l))

        acc1_h = self._xor(e14h, e11h)
        acc1_l = self._xor(e14l, e11l)
        log("acc1", (acc1_h, acc1_l))
        acc1_h, acc1_l = self._renorm_pair(acc1_h, acc1_l)

        acc2_h = self._xor(acc1_h, e13h)
        acc2_l = self._xor(acc1_l, e13l)
        log("acc2", (acc2_h, acc2_l))
        acc2_h, acc2_l = self._renorm_pair(acc2_h, acc2_l);

        out_h = self._xor(acc2_h, e9h);
        out_l = self._xor(acc2_l, e9l)
        out_h, out_l = self._renorm_pair(out_h, out_l)

        if do_final_bootstrap:
            out_h = eng.bootstrap(out_h);
            out_l = eng.bootstrap(out_l)
        log("out", (out_h, out_l))
        return out_h, out_l


# -------------------- 평문 레퍼런스 (column-first) --------------------
def _xtime_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint16)
    red = ((v & 0x80) != 0).astype(np.uint16) * 0x1B
    return (((v << 1) & 0xFF) ^ red).astype(np.uint8)


def _gf_mul_const_set(v: np.ndarray):
    """2,4,8 및 9/11/13/14를 한 번에 만들어 재사용"""
    m2 = _xtime_vec(v)
    m4 = _xtime_vec(m2)
    m8 = _xtime_vec(m4)
    m9 = (m8 ^ v).astype(np.uint8)  # 8 + 1
    m11 = (m8 ^ m2 ^ v).astype(np.uint8)  # 8 + 2 + 1
    m13 = (m8 ^ m4 ^ v).astype(np.uint8)  # 8 + 4 + 1
    m14 = (m8 ^ m4 ^ m2).astype(np.uint8)  # 8 + 4 + 2
    return m2, m4, m8, m9, m11, m13, m14


def _invmix_single_col_plain(col: np.ndarray) -> List[int]:
    """열 4바이트(col-major) 한 개에 대한 InvMixColumns (0e 0b 0d 09)"""
    a0, a1, a2, a3 = map(int, col)

    def gf_mul(a, b):
        res = 0
        for _ in range(8):
            if b & 1: res ^= a
            hi = a & 0x80
            a = (a << 1) & 0xFF
            if hi: a ^= 0x1B
            b >>= 1
        return res

    return [
        gf_mul(a0, 14) ^ gf_mul(a1, 11) ^ gf_mul(a2, 13) ^ gf_mul(a3, 9),
        gf_mul(a0, 9) ^ gf_mul(a1, 14) ^ gf_mul(a2, 11) ^ gf_mul(a3, 13),
        gf_mul(a0, 13) ^ gf_mul(a1, 9) ^ gf_mul(a2, 14) ^ gf_mul(a3, 11),
        gf_mul(a0, 11) ^ gf_mul(a1, 13) ^ gf_mul(a2, 9) ^ gf_mul(a3, 14),
    ]


def _invmixcolumns_plain(state16: np.ndarray) -> np.ndarray:
    """row-major 16바이트 입력 → row-major 16바이트 출력"""
    out = []
    # row-major에서 한 열은 (c, c+4, c+8, c+12)
    for c in range(4):
        col = state16[[c, c + 4, c + 8, c + 12]]
        out.extend(_invmix_single_col_plain(col))
    out = np.array(out, dtype=np.uint8)
    # 열별 결과(순차) → 다시 row-major로 배치
    rm = np.empty(16, dtype=np.uint8)
    for c in range(4):
        rm[[c, c + 4, c + 8, c + 12]] = out[c * 4:(c + 1) * 4]
    return rm


# -------------------- 테스트 (Inverse MixColumns) --------------------
if __name__ == "__main__":
    ctx = EngineContext(signature=1, max_level=17, mode="cpu", thread_count=4)
    enc = StateEncoder(ctx)

    xor4_coeffs = json.loads((COEFF_DIR / "xor4_coeffs.json").read_text(encoding="utf-8"))
    coeff_mat = np.zeros((16, 16), dtype=np.complex128)
    for p, q, re, im in xor4_coeffs["entries"]:
        coeff_mat[p, q] = complex(re, im)
    xor4 = XOR4LUT(ctx, coeff_mat)

    inv_mixc = InvMixColumnsFHE(ctx, xor4)

    # 입력
    np.random.seed(0)
    state = np.random.randint(0, 256, 16, dtype=np.uint8)
    ct_hi, ct_lo = enc.encode(state)

    # --- 평문 레퍼런스 (회전 기반 조합식) ---
    # 열 위로 1,2,3칸 회전 (row-major 기준, 각 열 독립 회전)
    r1_p = _rotate_rows_in_col_plain(state, 1)
    r2_p = _rotate_rows_in_col_plain(state, 2)
    r3_p = _rotate_rows_in_col_plain(state, 3)

    # 상수배 (9/11/13/14)
    _, _, _, nine_state, eleven_r1, thirteen_r2, fourteen_state = _gf_mul_const_set(state)
    _, _, _, nine_r3, _, _, _ = _gf_mul_const_set(r3_p)
    _, _, _, _, eleven_r1, _, _ = _gf_mul_const_set(r1_p)
    _, _, _, _, _, thirteen_r2, _ = _gf_mul_const_set(r2_p)
    fourteen_state = _gf_mul_const_set(state)[-1]  # 14*state

    # 조합식: InvMix = 14*state ⊕ 11*r1 ⊕ 13*r2 ⊕ 9*r3
    acc1_p = (fourteen_state ^ eleven_r1).astype(np.uint8)
    acc2_p = (acc1_p ^ thirteen_r2).astype(np.uint8)
    acc3_p = (acc2_p ^ nine_r3).astype(np.uint8)

    # 정식 행렬 버전(크로스체크)
    invmix_p = _invmixcolumns_plain(state)

    # --- 실행 + 비교 ---
    dbg: Dict[str, Any] = {}
    t0 = time.perf_counter()
    out_hi, out_lo = inv_mixc(ct_hi, ct_lo, do_final_bootstrap=True, debug=dbg)
    t1 = time.perf_counter()

    # 입력 검증
    _print_cmp("input", enc.decode(ct_hi, ct_lo), state)

    print_cmp_safe("rotc1", dec_pair_safe("rotc1", dbg, enc), r1_p)
    print_cmp_safe("rotc2", dec_pair_safe("rotc2", dbg, enc), r2_p)
    print_cmp_safe("rotc3", dec_pair_safe("rotc3", dbg, enc), r3_p)

    print_cmp_safe("mul14", dec_pair_safe("mul14", dbg, enc), fourteen_state)
    print_cmp_safe("mul11@r1", dec_pair_safe("mul11", dbg, enc), eleven_r1)
    print_cmp_safe("mul13@r2", dec_pair_safe("mul13", dbg, enc), thirteen_r2)
    print_cmp_safe("mul9@r3", dec_pair_safe("mul9", dbg, enc), nine_r3)

    # XOR 누적 단계 비교 (네 구현에서 dbg 키: acc1/acc2/acc3로 맞춰두면 바로 비교됨)
    print_cmp_safe("acc1", dec_pair_safe("acc1", dbg, enc), acc1_p)
    print_cmp_safe("acc2", dec_pair_safe("acc2", dbg, enc), acc2_p)
    # print_cmp_safe("acc3", dec_pair_safe("acc3", dbg, enc), acc3_p)

    # 최종 비교: 두 가지 레퍼런스 모두와 매칭 체크
    dec_out = enc.decode(out_hi, out_lo)
    _print_cmp("final (vs acc3_p)", dec_out, acc3_p)
    _print_cmp("final (vs invmix)", dec_out, invmix_p)

    print(f"Total time: {t1 - t0:.3f}s")
