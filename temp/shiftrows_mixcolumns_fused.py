"""
- input : SubBytes 이후의 ct_hi, ct_lo
- lo-lift(Zeta16 -> Zeta256)로 바이트 도메인 복원 -> ct_b = ct_hi * lift(ct_lo)
- ShiftRows는 row-mask * row-rotation(±4*stride)로 구현
- MixColumns(encryption)는 ×2, ×3 (8->8 LUT)를 ct_b에 평가한 뒤
  행별 값들을 '목표 행' 슬롯로 정렬·수집하여 4→4 XOR 체인으로 합산
- 다음 단계가 4-bit XOR(AddRoundKey)이면, 이미 니블 도메인(ζ16)이라 바로 연결


attempt to GHS12 implementation in 2024-274

"""
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from engine_context import EngineContext
from state_encoder import StateEncoder
from xor4_lut import XOR4LUT


# ---------------- utilities to build 8->8 LUT coeffs ----------------
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


# ---------------- fused SR+MC (encryption) ----------------
class ShiftRowsMixColumnsFusedEnc:
    def __init__(self,
                 ctx: EngineContext,
                 split_hi_coeff: np.ndarray,
                 split_lo_coeff: np.ndarray,
                 xor4: XOR4LUT):
        self.ctx = ctx
        self.sc = ctx.engine.slot_count
        self.stride = self.sc // 16
        self.xor4 = xor4

        # ---- row masks (row r indices : r + 4*c) ----
        self.pt_row: List[Any] = []
        for r in range(4):
            m = np.zeros(self.sc, dtype=np.complex128)
            for c in range(4):
                m[(r + 4 * c) * self.stride] = 1.0
            self.pt_row.append(ctx.encode(m))
        # ShiftRows 회전 스텝 (왼쪽 shift r칸)
        self.rot_row = [-r * 4 * self.stride for r in range(4)]

        # ---- 8->8 LUT coeffs for ×2, ×3 (encryption) ----
        c2 = _ifft_coeff_from_table(_gf_mul_table(2))
        c3 = _ifft_coeff_from_table(_gf_mul_table(3))
        tol = 1e-12
        self.ks2 = [k for k, v in enumerate(c2) if abs(v) > tol]
        self.ks3 = [k for k, v in enumerate(c3) if abs(v) > tol]
        self.deg256 = min(max(self.ks2 + self.ks3) if (self.ks2 or self.ks3) else 0, 128)

        # plaintext-encoded coeffs (성능)
        self.pt_c2: Dict[int, Any] = {k: ctx.encode(np.full(self.sc, c2[k], dtype=np.complex128)) for k in self.ks2}
        self.pt_c3: Dict[int, Any] = {k: ctx.encode(np.full(self.sc, c3[k], dtype=np.complex128)) for k in self.ks3}
        self.c20 = c2[0]
        self.c30 = c3[0]

        # ---- 16->256 lo-lift: Zeta16^lo -> Zeta256^lo ----
        z256 = np.exp(-2j * np.pi / 256)
        lift = np.fft.ifft(np.array([z256 ** k for k in range(16)], dtype=np.complex128))
        self.ks_lift = [k for k, v in enumerate(lift) if abs(v) > tol and k != 0]
        self.deg16 = min(max(self.ks_lift) if self.ks_lift else 0, 8)
        self.pt_lift = {k: ctx.encode(np.full(self.sc, lift[k], dtype=np.complex128)) for k in self.ks_lift}
        self.c0_lift = lift[0]

        # ---- optional 8->4 split LUT (256->16) — 여기선 사용하지 않음 (니블로 돌아오지 않음) ----
        # 다음 단계(ARK)를 4-bit XOR로 하므로, 결과를 니블 도메인으로 유지할 거라 split은 호출하지 않음.
        self.split_hi = split_hi_coeff
        self.split_lo = split_lo_coeff

    # ---- helpers ----
    def _apply_poly_pt(self, ct: Any, pt_coeff: Dict[int, Any], c0: complex,
                       domain: int, deg: int) -> Any:
        """
        1D LUT 평가: coeff[k]*(ct^k)를 합산 (NTT 상태면 bootstrap 후 재시도)
        """
        eng = self.ctx
        res = eng.add_plain(eng.multiply(ct, 0.0), c0)
        if deg <= 0:
            return res
        try:
            pos = eng.make_power_basis(ct, deg)
        except RuntimeError as e:
            if "NTT" in str(e) or "positive level" in str(e):
                ct = eng.bootstrap(ct)
                pos = eng.make_power_basis(ct, deg)
            else:
                raise
        for k, pt in pt_coeff.items():
            bk = pos[k - 1] if k <= deg else eng.conjugate(pos[domain - k - 1])
            res = eng.add(res, eng.multiply(bk, pt))
        return res

    def _lift_lo(self, ct_lo: Any) -> Any:
        """
        Zeta16^lo -> Zeta256^lo (NTT 안전)
        """
        eng = self.ctx
        res = eng.add_plain(eng.multiply(ct_lo, 0.0), self.c0_lift)
        if self.deg16 <= 0:
            return res
        try:
            pos16 = eng.make_power_basis(ct_lo, self.deg16)
        except RuntimeError as e:
            if "NTT" in str(e) or "positive level" in str(e):
                ct_lo = eng.bootstrap(ct_lo)
                pos16 = eng.make_power_basis(ct_lo, self.deg16)
            else:
                raise
        for k, pt in self.pt_lift.items():
            bk = pos16[k - 1] if k <= self.deg16 else eng.conjugate(pos16[16 - k - 1])
            res = eng.add(res, eng.multiply(bk, pt))
        return res

    def _row_shift(self, x):
        """
        ShiftRows: 각 행 r을 왼쪽으로 r칸 (열 간 간격은 4*stride)
        """
        eng = self.ctx
        parts = []
        for r in range(4):
            masked = eng.multiply(x, self.pt_row[r])              # 행 r만 남기고
            parts.append(eng.rotate(masked, self.rot_row[r]))     # 왼쪽 r칸 회전
        out = eng.multiply(x, 0.0)
        for p in parts:
            out = eng.add(out, p)
        return out

    def _align_to_row(self, ct, src_r, dst_r):
        """
        입력 행(src_r)의 값을 같은 열 인덱스를 유지한 채 출력 행(dst_r)의 슬롯 위치로 정렬
        """
        eng = self.ctx
        off = (dst_r - src_r) * self.stride
        return eng.rotate(eng.multiply(ct, self.pt_row[src_r]), off)

    # ---- main ----
    def __call__(self, ct_hi, ct_lo, return_split16: bool = True):
        eng = self.ctx

        # 1) ζ16 → ζ256
        lo_lift = self._lift_lo(ct_lo)
        ct_b = eng.multiply(ct_hi, lo_lift)  # 바이트 도메인

        # 2) ShiftRows
        A = self._row_shift(ct_b)  # A' = ShiftRows(A)

        try:
            A = eng.bootstrap(A)
        except Exception:
            pass

        # 3) ×1, ×2, ×3 (8→8 LUT)
        ct1 = A
        ct2 = self._apply_poly_pt(A, self.pt_c2, self.c20, 256, self.deg256)
        ct3 = self._apply_poly_pt(A, self.pt_c3, self.c30, 256, self.deg256)

        # 4) 256→16 split (저심도): lo = x^16, hi = x * P(x^16)
        def split_low_depth(x_ct):
            # x16 = x^16 via 4 squarings
            def _sqr(y):
                z = eng.multiply(y, y)
                return eng.relinearize(z)
            try:
                x2 = _sqr(x_ct); x4 = _sqr(x2); x8 = _sqr(x4); x16 = _sqr(x8)
            except RuntimeError as e:
                # 레벨/NTT 이슈 시 한 번 refresh
                x_ct = eng.bootstrap(x_ct)
                x2 = _sqr(x_ct); x4 = _sqr(x2); x8 = _sqr(x4); x16 = _sqr(x8)

            # P(z) = Σ_m a[m] z^m, where a[m] = split_hi[16*m+1]
            a = np.zeros(16, dtype=np.complex128)
            if self.split_hi is not None and len(self.split_hi) >= 256:
                for m in range(16):
                    k = 16 * m + 1
                    a[m] = complex(self.split_hi[k])
            # basis for domain 16
            try:
                pos = eng.make_power_basis(x16, 8)
            except RuntimeError as e:
                if "NTT" in str(e) or "positive level" in str(e):
                    x16 = eng.bootstrap(x16)
                    pos = eng.make_power_basis(x16, 8)
                else:
                    raise
            basis = {0: eng.add_plain(x16, 1.0)}
            for m in range(1, 16):
                basis[m] = pos[m - 1] if m <= 8 else eng.conjugate(pos[16 - m - 1])

            P = eng.multiply(x16, 0.0)
            tol = 1e-12
            if abs(a[0]) > tol:
                P = eng.add_plain(P, a[0])
            for m in range(1, 16):
                if abs(a[m]) < tol:
                    continue
                P = eng.add(P, eng.multiply_plain(basis[m], a[m]))

            hi16 = eng.relinearize(eng.multiply(x_ct, P))
            lo16 = x16
            return hi16, lo16

        h1, l1 = split_low_depth(ct1)
        h2, l2 = split_low_depth(ct2)
        h3, l3 = split_low_depth(ct3)

        # 5) 행 정렬 후 4→4 XOR로 합산 (각 행은 서로 겹치지 않음)
        # Row0: [2,3,1,1] from rows [0,1,2,3]
        r0_h = self.xor4(self.xor4(self._align_to_row(h2, 0, 0), self._align_to_row(h3, 1, 0)),
                         self.xor4(self._align_to_row(h1, 2, 0), self._align_to_row(h1, 3, 0)))
        r0_l = self.xor4(self.xor4(self._align_to_row(l2, 0, 0), self._align_to_row(l3, 1, 0)),
                         self.xor4(self._align_to_row(l1, 2, 0), self._align_to_row(l1, 3, 0)))

        # Row1: [1,2,3,1]
        r1_h = self.xor4(self.xor4(self._align_to_row(h1, 0, 1), self._align_to_row(h2, 1, 1)),
                         self.xor4(self._align_to_row(h3, 2, 1), self._align_to_row(h1, 3, 1)))
        r1_l = self.xor4(self.xor4(self._align_to_row(l1, 0, 1), self._align_to_row(l2, 1, 1)),
                         self.xor4(self._align_to_row(l3, 2, 1), self._align_to_row(l1, 3, 1)))

        # Row2: [1,1,2,3]
        r2_h = self.xor4(self.xor4(self._align_to_row(h1, 0, 2), self._align_to_row(h1, 1, 2)),
                         self.xor4(self._align_to_row(h2, 2, 2), self._align_to_row(h3, 3, 2)))
        r2_l = self.xor4(self.xor4(self._align_to_row(l1, 0, 2), self._align_to_row(l1, 1, 2)),
                         self.xor4(self._align_to_row(l2, 2, 2), self._align_to_row(l3, 3, 2)))

        # Row3: [3,1,1,2]
        r3_h = self.xor4(self.xor4(self._align_to_row(h3, 0, 3), self._align_to_row(h1, 1, 3)),
                         self.xor4(self._align_to_row(h1, 2, 3), self._align_to_row(h2, 3, 3)))
        r3_l = self.xor4(self.xor4(self._align_to_row(l3, 0, 3), self._align_to_row(l1, 1, 3)),
                         self.xor4(self._align_to_row(l1, 2, 3), self._align_to_row(l2, 3, 3)))

        # 6) 행별 결과 합치기 (서로 슬롯 불겹침이므로 add)
        out_hi = eng.add(eng.add(r0_h, r1_h), eng.add(r2_h, r3_h))
        out_lo = eng.add(eng.add(r0_l, r1_l), eng.add(r2_l, r3_l))

        # 이미 니블 도메인(ζ16): 바로 ARK(4-bit XOR)로 진행 가능
        return out_hi, out_lo


# -------- test harness ----------
from lut import load_coeff1d, load_coeff2d
import time


def main():
    # EngineContext가 signature=2를 지원하지 않으면 1로 바꿔 실행해
    ctx = EngineContext(signature=1,use_bootstrap=True, max_level=17, mode='cpu', thread_count=4)
    encoder = StateEncoder(ctx)

    base = Path(__file__).parent / "generator" / "coeffs"
    split_hi = load_coeff1d(base / "split_mod256_to_16_hi.json")
    split_lo = load_coeff1d(base / "split_mod256_to_16_lo.json")
    xor4_coeffs = load_coeff2d(base / "xor4_coeffs.json", 16)
    xor4 = XOR4LUT(ctx, xor4_coeffs)

    fused = ShiftRowsMixColumnsFusedEnc(ctx, split_hi, split_lo, xor4)

    # test data
    np.random.seed(0)
    state = np.random.randint(0, 256, 16, dtype=np.uint8)
    ct_hi, ct_lo = encoder.encode(state)

    t0 = time.perf_counter()
    out_hi, out_lo = fused(ct_hi, ct_lo, return_split16=True)
    t1 = time.perf_counter()
    dec = encoder.decode(out_hi, out_lo)

    # 평문 체크
    def shiftrows_plain_colfirst(x):
        M = np.asarray(x, dtype=np.uint8).reshape(4, 4, order='F')
        for r in range(4):
            M[r, :] = np.roll(M[r, :], -r)
        return M.reshape(16, order='F')

    def mix_single_col(col):
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
        a0, a1, a2, a3 = col
        return [
            gf_mul(a0, 2) ^ gf_mul(a1, 3) ^ a2 ^ a3,
            a0 ^ gf_mul(a1, 2) ^ gf_mul(a2, 3) ^ a3,
            a0 ^ a1 ^ gf_mul(a2, 2) ^ gf_mul(a3, 3),
            gf_mul(a0, 3) ^ a1 ^ a2 ^ gf_mul(a3, 2),
        ]

    def mixcolumns_plain(x):
        out = []
        for c in range(4):
            col = x[c * 4:(c + 1) * 4]
            out.extend(mix_single_col(col))
        return np.array(out, dtype=np.uint8)

    exp = mixcolumns_plain(shiftrows_plain_colfirst(state))
    print("State           :", state)
    print("MIXED & SHIFT   :", dec)
    print("Expected        :", exp)
    print("Match?          :", np.array_equal(dec, exp))
    print(f"SR+MC time      : {t1 - t0: .6f} s")


if __name__ == "__main__":
    main()