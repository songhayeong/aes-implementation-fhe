from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple, List
import json
import numpy as np

from engine_context import EngineContext
from lut import load_coeff2d
from state_encoder import StateEncoder
from xor4_lut import XOR4LUT

COEFF_DIR = Path(__file__).parent / "generator" / "coeffs"


# -------------------- Coeff cache --------------------
class _CoeffCache:
    def __init__(self):
        self.pt_cache: Dict[Tuple[int, str], Dict[Tuple[int, int], Any]] = {}

    def load_plaintexts(self, ctx: EngineContext, mult: int, which: str) -> Dict[Tuple[int, int], Any]:
        """
        coeff json: {"entries": [[p, q, re, im], ...]}
        → {(p,q): Plaintext(encoded constant vector)}
        """
        key = (mult, which)
        if key in self.pt_cache:
            return self.pt_cache[key]
        path = COEFF_DIR / f"gf_mult{mult}_{which}_coeffs.json"
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        entries = obj["entries"]
        slot = ctx.engine.slot_count

        d: Dict[Tuple[int, int], Any] = {}
        for p, q, re, im in entries:
            const = np.full(slot, complex(re, im), dtype=np.complex128)
            d[(p, q)] = ctx.encode(const)
        self.pt_cache[key] = d
        return d


# -------------------- MixColumns (row-in-col rotation) --------------------
class MixColumnsDesiloPort:
    """
    입력/출력: ζ16 니블 도메인 (ct_hi, ct_lo) -> (out_hi, out_lo)
    내부: (ct_hi, ct_lo)에 대해 2-변수 폴리(니블)로 ×2, ×3 평가 후
         '열-내 행 회전(r=-1,-2,-3)' 결과들과 XOR 누적.
    """

    def __init__(self, ctx: EngineContext, xor4: XOR4LUT, stride: int | None = None):
        self.ctx = ctx
        self.xor4 = xor4
        self.sc = ctx.engine.slot_count
        self.stride = stride if stride is not None else (self.sc // 16)
        self._coeffs = _CoeffCache()

        # 행 마스크(4개): column-first에서 (r + 4*c)*stride.. 의 구간을 1로
        self.pt_block = []
        for b in range(16):
            m = np.zeros(self.sc, dtype=np.complex128)
            start = b * self.stride
            m[start:start + self.stride] = 1.0
            self.pt_block.append(self.ctx.encode(m))

        self.pt_col_rm = []
        for c in range(4):
            m = np.zeros(self.sc, dtype=np.complex128)
            for r in range(4):
                idx = r * 4 + c  # row - major 에서 (r, c)의 인덱스
                start = idx * self.stride
                m[start:start + self.stride] = 1.0  # 그 요소 블록만 1
            self.pt_col_rm.append(self.ctx.encode(m))

    def _rot_rows_in_col_strict(self, ct: Any, k_rows: int) -> Any:
        """
        컬럼 내부에서만 행 회전 (r -> (r+k) mod 4 in same column).
        블록(16개)별로 잘라서, 목적 블록으로만 이동시킨 뒤 합침.
        """
        eng = self.ctx
        out = eng.multiply(ct, 0.0)  # zero 같은 레벨로
        for c in range(4):
            for r in range(4):
                src_b = r + 4 * c
                dst_b = ((r + k_rows) % 4) + 4 * c
                delta = (dst_b - src_b) * self.stride  # 블록 차이만큼 회전
                part = eng.multiply(ct, self.pt_block[src_b])  # (r,c)만 남김
                part = eng.rotate(part, delta)  # 목적 블록으로 이동
                out = eng.add(out, part)
        return out

    def _col_shift_rowmajor(self, ct: Any, k_up: int) -> Any:
        """
        row - major 4*4에서 '각 열을 위로 K칸' == 행 블록을 위로 k칸
         = > 평탄화 벡터를 -4*k 요소만큼 한번에 회전
        """
        return self.ctx.rotate(ct, -k_up * 4 * self.stride)

    def _col_shift_rowmajor_masked(self, ct: Any, k_up: int) -> Any:
        """
        backup root (마스크 분해) : 각 열만 남겨서 -4*k 회전 후 합산
        """
        eng = self.ctx
        parts = []
        for c in range(4):
            masked = eng.multiply(ct, self.pt_col_rm[c])
            parts.append(eng.rotate((masked, -k_up * 4 * self.stride)))
        out = eng.multiply(ct, 0.0)
        for p in parts:
            out = eng.add(out, p)
        return out

    # ζ16 power basis (0..15). 1..8은 power, 9..15는 켤레로 보충
    def _basis16(self, ct: Any) -> Dict[int, Any]:
        eng = self.ctx
        try:
            pos = eng.make_power_basis(ct, 8)
        except RuntimeError:
            ct = eng.bootstrap(ct)
            pos = eng.make_power_basis(ct, 8)
        # zero = eng.multiply(ct, 0.0)
        zero = eng.sub(ct, ct)
        basis0 = eng.add_plain(zero, 1.0)   # 어기도 그냥 바로 ciphertext에서 더하게 하면 될듯
        basis = {0: basis0}
        for k in range(1, 9):
            basis[k] = pos[k - 1]
        for k in range(9, 16):
            basis[k] = eng.conjugate(pos[(16 - k) - 1])
        return basis

    # Σ c[p,q] * X^p * Y^q  (coeff는 이미 Plaintext)
    def _gf_poly_eval_2var(self, ct_hi: Any, ct_lo: Any, mult: int, which: str) -> Any:
        eng = self.ctx
        bx = self._basis16(ct_hi)
        by = self._basis16(ct_lo)
        coeff_pt = self._coeffs.load_plaintexts(eng, mult, which)

        res = eng.multiply(ct_hi, 0.0)  # zero ct (same level/scale)
        for (p, q), pt in coeff_pt.items():
            term = eng.multiply(bx[p], by[q])  # ct * ct
            term = eng.multiply(term, pt)  # * PT
            res = eng.add(res, term)
        return res

    def gf_mult_2(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return (self._gf_poly_eval_2var(ct_hi, ct_lo, 2, "hi"),
                self._gf_poly_eval_2var(ct_hi, ct_lo, 2, "lo"))

    def gf_mult_3(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return (self._gf_poly_eval_2var(ct_hi, ct_lo, 3, "hi"),
                self._gf_poly_eval_2var(ct_hi, ct_lo, 3, "lo"))

    # ★ 열-내 행 회전: 각 행을 마스크로 추출 → 동일 stride만큼 회전 → 합산
    def _rot_rows_in_col(self, ct: Any, k_rows: int) -> Any:
        eng = self.ctx
        parts = []
        for r in range(4):
            masked = eng.multiply(ct, self.pt_row[r])  # 행 r만 남김
            parts.append(eng.rotate(masked, k_rows * self.stride))
        out = eng.multiply(ct, 0.0)
        for p in parts:
            out = eng.add(out, p)
        return out

    def _ensure_xor_ready(self, ct: Any) -> Any:
        """
        XOR LUT 전에 최소 레벨/스케일 확인. 부족하면 bootstrap.
        (여기선 단순: power_basis(1) 시도 → 실패 시 bootstrap)
        """
        try:
            # self.ctx.make_power_basis(ct, 1)
            return ct
        except RuntimeError:
            return self.ctx.bootstrap(ct)

    def _to_coeff(self, ct):
        # NTT로 올려져 있으면 coefficient = INTT로 강제
        try:
            return self.ctx.to_intt(ct)
        except Exception:
            return ct

    def _xor_ct(self, a, b):
        """
        XOR4에 넣기 전 둘 다 coefficient로 맞춤
        """
        a = self._to_coeff(a) # ct
        b = self._to_coeff(b) # ct
        return self.xor4(self._ensure_xor_ready(a), self._ensure_xor_ready(b))

    def __call__(self, ct_hi: Any, ct_lo: Any,
                 do_final_bootstrap: bool = True,
                 debug: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        eng = self.ctx
        if debug is not None:
            debug.clear()

            def dbg(k, v):
                debug[k] = v
        else:
            def dbg(k, v):
                pass

        # (1) 열-내 행 회전 준비: -1, -2, -3
        r1_hi, r1_lo = self._col_shift_rowmajor(ct_hi, 1), self._col_shift_rowmajor(ct_lo, 1)
        r2_hi, r2_lo = self._col_shift_rowmajor(ct_hi, 2), self._col_shift_rowmajor(ct_lo, 2)
        r3_hi, r3_lo = self._col_shift_rowmajor(ct_hi, 3), self._col_shift_rowmajor(ct_lo, 3)
        dbg("rotc1", (r1_hi, r1_lo))
        dbg("rotc2", (r2_hi, r2_lo))
        dbg("rotc3", (r3_hi, r3_lo))
        dbg("in", (ct_hi, ct_lo))

        # (2) GF ×2 (orig), ×3 (r1)
        two_hi, two_lo = self.gf_mult_2(ct_hi, ct_lo)
        thr_hi, thr_lo = self.gf_mult_3(r1_hi, r1_lo)
        dbg("two", (two_hi, two_lo))
        dbg("thr", (thr_hi, thr_lo))

        # 잠깐만 원래 LUT는 4bit * 4bit xor에 대해 한거일텐대 (gf2*df3)의 곱과 기존 bit 체계에서의 곱은 또 그러면 새롭게 LUT를 정의해야하는거 아닌가 ?
        # 그런 LUT를 정의안한 상태에서 XOR을 하려해서 생긴 문제인건가 ? 원래 LUT는 4bit * 4bit xor에 대해 한거일텐대
        # 그렇다고 하기엔 LUT는 모든 값들을 다 들고 있을텐데...

        # 3) XOR 누적 — 회전본은 coefficient 도메인으로만 맞춰서 XOR
        acc1_hi = self._xor_ct(two_hi, thr_hi)
        acc1_lo = self._xor_ct(two_lo, thr_lo)
        dbg("acc1", (acc1_hi, acc1_lo))

        # ---- XOR 불변성 / 대칭성 체크 (acc2 원인 규명용)
        acc1_hi_n = self._to_coeff(acc1_hi)
        r2_hi_n = self._to_coeff(r2_hi)
        acc1_lo_n = self._to_coeff(acc1_lo)
        r2_lo_n = self._to_coeff(r2_lo)

        z_hi = self._xor_ct(acc1_hi_n, acc1_hi_n)
        z_lo = self._xor_ct(acc1_lo_n, acc1_lo_n)
        dbg("xor_a^a", (z_hi, z_lo))    # 실제로 봤을때 같은것에 대해 xor을 한다면 정확히 0이 나와야함 근데
                                        # debug를 찍어보면 0이 나오지않고 255 즉 = 1에 가까운 숫자들이 대거나옴
                                        # 이는 xor 관련 부분에 문제가 있음을 시사.
                                        # 하지만 이전 xor에서는 통과함.
        # B) 회전끼리 XOR -> 평문 r2^r3와 일치해야 한다 (둘 다 회전 경로)
        t_hi = self._xor_ct(r2_hi_n, r3_hi)
        t_lo = self._xor_ct(r2_lo_n, r3_lo)
        dbg("xor_r2^r3:", (t_hi, t_lo))

        # (C) 순서 뒤집기 테스트 (비대칭/스케일 이슈 구분)
        acc2_hi_fwd = self._xor_ct(acc1_hi_n, r2_hi_n)
        acc2_hi_rev = self._xor_ct(r2_hi_n, acc1_hi_n)
        acc2_lo_fwd = self._xor_ct(acc1_lo_n, r2_lo_n)
        acc2_lo_rev = self._xor_ct(r2_lo_n, acc1_lo_n)
        dbg("acc2_fwd", (acc2_hi_fwd, acc2_lo_fwd))
        dbg("acc2_rev", (acc2_hi_rev, acc2_lo_rev))
        # --- 여기까지 ---

        acc2_hi = self._xor_ct(acc1_hi, r2_hi)  # r2_hi 내부에서 to_intt 적용됨
        acc2_lo = self._xor_ct(acc1_lo, r2_lo)
        dbg("acc2", (acc2_hi, acc2_lo))

        acc3_hi = self._xor_ct(acc2_hi, r3_hi)
        acc3_lo = self._xor_ct(acc2_lo, r3_lo)
        dbg("acc3", (acc3_hi, acc3_lo))

        out_hi, out_lo = acc3_hi, acc3_lo
        if do_final_bootstrap:
            out_hi = self.ctx.bootstrap(out_hi);
            out_lo = self.ctx.bootstrap(out_lo)
        dbg("out", (out_hi, out_lo))
        return out_hi, out_lo


# --- helpers for plain expected values (column-first, per-column row rotation) ---
def _rotate_rows_in_col_plain(vec_cf: np.ndarray, k_rows: int) -> np.ndarray:
    """
    column-first 벡터(길이 16)에서 같은 '열' 내부(4바이트)만 k_rows만큼 회전.
    k_rows = -1 이면 '위로 1칸' ( [r0,r1,r2,r3] -> [r1,r2,r3,r0] ).
    """
    k = k_rows % 4
    out = vec_cf.copy()
    for c in range(4):
        s = 4 * c
        # 위로 k_rows칸: numpy는 음수가 왼쪽(=앞으로)이므로 -k_rows가 아니라 k_rows 그대로 쓰되 부호 일관 유지
        out[s:s + 4] = np.roll(out[s:s + 4], k_rows)
    return out


def _gf_mul2_vec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint16)
    x = (v << 1)
    red = ((v & 0x80) != 0).astype(np.uint16) * 0x1B
    return ((x & 0xFF) ^ red).astype(np.uint8)


def _gf_mul3_vec(v: np.ndarray) -> np.ndarray:
    return (_gf_mul2_vec(v) ^ v).astype(np.uint8)


def _mix_single_col_plain(col: np.ndarray) -> list[int]:
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


def rotate_columns_rowmajor_plain(state16: np.ndarray, k_up: int) -> np.ndarray:
    M = state16.reshape(4, 4)
    M2 = np.roll(M, -k_up, axis=0)
    return M2.reshape(-1)


def _mixcolumns_plain(state_cf: np.ndarray) -> np.ndarray:
    out = []
    for c in range(4):
        col = state_cf[c * 4:(c + 1) * 4]
        out.extend(_mix_single_col_plain(col))
    return np.array(out, dtype=np.uint8)


def _print_cmp(tag: str, got: np.ndarray, exp: np.ndarray):
    ok = np.array_equal(got, exp)
    print(f"[{tag:10s}] OK? {ok} | got={got.tolist()} | exp={exp.tolist()}")
    if not ok:
        bad = np.where(got != exp)[0]
        print(f"  mismatch idx: {bad.tolist()}")
        print("  pairs:", [(int(got[i]), int(exp[i])) for i in bad])


def dec_pair_safe(name: str, dbg: Dict[str, Any], enc: StateEncoder):
    if name not in dbg:
        print(f"[WARN] debug['{name}'] 없음. keys={sorted(dbg.keys())}")
        return None
    try:
        h, l = dbg[name]
        return enc.decode(h, l)
    except Exception as e:
        print(f"[ERR] decode '{name}': {e!r}")
        return None


def print_cmp_safe(tag: str, got: np.ndarray | None, exp: np.ndarray):
    if got is None:
        print(f"[{tag:10s}] SKIP (got=None)")
        return
    _print_cmp(tag, got, exp)


# -------------------- main --------------------
if __name__ == "__main__":
    ctx = EngineContext(signature=1, max_level=17, mode="cpu", thread_count=4)
    enc = StateEncoder(ctx)

    xor4_coeffs = load_coeff2d(Path("generator/coeffs/xor4_coeffs.json"), 16)
    xor4 = XOR4LUT(ctx, xor4_coeffs)
    mixc = MixColumnsDesiloPort(ctx, xor4)

    # 랜덤 입력
    np.random.seed(0)
    state = np.random.randint(0, 256, 16, dtype=np.uint8)  # ※ column-first 벡터로 취급

    # 니블 인코딩(암호문)
    ct_hi, ct_lo = enc.encode(state)

    # (1) 기대 회전값: 열 내부 행 회전 (column-first 블록별 4칸 중에서만 회전)
    r1_p = rotate_columns_rowmajor_plain(state, 1)  # 위로 1칸
    r2_p = rotate_columns_rowmajor_plain(state, 2)  # 위로 2칸
    r3_p = rotate_columns_rowmajor_plain(state, 3)  # 위로 3칸

    # (2) 기대 GF곱 및 XOR 누적
    two_p = _gf_mul2_vec(state)  # 2 * orig
    thr_p = _gf_mul3_vec(r1_p)  # 3 * rot1
    acc1_p = (two_p ^ thr_p).astype(np.uint8)   # 여기까지 xor같은 루틴을 씀.
    acc2_p = (acc1_p ^ r2_p).astype(np.uint8)   # (gf(2) * rot0 ^ gf(3)rot1) ^ rot2
    acc3_p = (acc2_p ^ r3_p).astype(np.uint8)
    mix_p = _mixcolumns_plain(state)  # 표준 MixColumns 레퍼런스

    # 실행 + 디버그
    dbg: Dict[str, Any] = {}
    t0 = time.perf_counter()
    out_hi, out_lo = mixc(ct_hi, ct_lo, do_final_bootstrap=True, debug=dbg)
    t1 = time.perf_counter()

    # 단계별 비교 (엔진이 기록한 디버그 키: rotc1/rotc2/rotc3, two, thr, acc1/acc2/acc3)
    _print_cmp("input", enc.decode(ct_hi, ct_lo), state)
    print_cmp_safe("rotc1", dec_pair_safe("rotc1", dbg, enc), r1_p)
    print_cmp_safe("rotc2", dec_pair_safe("rotc2", dbg, enc), r2_p)
    print_cmp_safe("rotc3", dec_pair_safe("rotc3", dbg, enc), r3_p)
    print_cmp_safe("gf*2", dec_pair_safe("two", dbg, enc), two_p)
    print_cmp_safe("gf*3@c1", dec_pair_safe("thr", dbg, enc), thr_p)
    print_cmp_safe("acc1", dec_pair_safe("acc1", dbg, enc), acc1_p)
    print_cmp_safe("acc2", dec_pair_safe("acc2", dbg, enc), acc2_p)
    print_cmp_safe("acc3", dec_pair_safe("acc3", dbg, enc), acc3_p)

    # acc2/acc3에 대해 “LUT 문제인지” 빠르게 구분하는 테스트(평문 XOR vs FHE 출력)도 같이 표기
    dec_acc1 = dec_pair_safe("acc1", dbg, enc)
    dec_r2 = dec_pair_safe("rotc2", dbg, enc)
    acc2_fhe = dec_pair_safe("acc2", dbg, enc)
    if dec_acc1 is not None and dec_r2 is not None and acc2_fhe is not None:
        acc2_numpy = np.bitwise_xor(dec_acc1.astype(np.uint8), dec_r2.astype(np.uint8))
        print_cmp_safe("acc2_numpy_vs_fhe", acc2_numpy, acc2_fhe)

    dec_acc2 = dec_pair_safe("acc2", dbg, enc)
    dec_r3 = dec_pair_safe("rotc3", dbg, enc)
    acc3_fhe = dec_pair_safe("acc3", dbg, enc)
    if dec_acc2 is not None and dec_r3 is not None and acc3_fhe is not None:
        acc3_numpy = np.bitwise_xor(dec_acc2.astype(np.uint8), dec_r3.astype(np.uint8))
        print_cmp_safe("acc3_numpy_vs_fhe", acc3_numpy, acc3_fhe)

    # 0 벡터 / 대칭성 / 회전끼리 XOR 검증
    dec_xora = dec_pair_safe("xor_a^a", dbg, enc)  # 전부 0이어야 함
    if dec_xora is not None:
        zeros = np.zeros_like(state, dtype=np.uint8)
        _print_cmp("xor(a,a)==0", dec_xora, zeros)

    dec_r2r3 = dec_pair_safe("xor_r2^r3", dbg, enc)
    if dec_r2r3 is not None:
        r2_p = _rotate_rows_in_col_plain(state, -2)  # 너의 plain 기준
        r3_p = _rotate_rows_in_col_plain(state, -3)
        _print_cmp("xor(r2,r3)", dec_r2r3, (r2_p ^ r3_p).astype(np.uint8))

    rep = (((state >> 4) ^ (state & 0x0F)) & 0x0F)
    rep = ((rep << 4) | rep).astype(np.uint8)
    dec_xora = dec_pair_safe("xor_a^a", dbg, enc)  # 이미 만들어둔 디버그 키
    if dec_xora is not None:
        _print_cmp("xor(a,a) vs 0x11*(hi^lo)", dec_xora, rep)

    dec_out = enc.decode(out_hi, out_lo)
    _print_cmp("final", dec_out, mix_p)
    print(f"Total time: {t1 - t0:.3f}s")
