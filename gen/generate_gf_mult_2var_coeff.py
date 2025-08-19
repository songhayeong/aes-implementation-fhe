# gen_gf_mult_2var_coeffs.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# ===== 저장 경로 =====
OUT_DIR = Path(__file__).parent / "coeffs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== GF(2^8) 기본 곱셈 =====
REDU_POLY = 0x1B  # x^8 + x^4 + x^3 + x + 1 (AES)


def gf_mul_const(x: int, k: int) -> int:
    """AES GF(2^8)에서 바이트 x(0..255)에 상수 k(정수) 곱."""
    a = x
    b = k
    res = 0
    for _ in range(8):
        if b & 1:
            res ^= a
        hi = a & 0x80
        a = ((a << 1) & 0xFF)
        if hi:
            a ^= REDU_POLY
        b >>= 1
    return res


# ===== 2D IFFT로 c[p,q] 뽑기 =====
def two_var_coeffs_for_multiplier(k: int, which: str, tol: float = 1e-12):
    """
    k ∈ {2,3,...}, which ∈ {"hi","lo"}
    반환: entries = [(p,q,re,im), ...]  (희소화)
    """
    z16 = np.exp(-2j * np.pi / 16)  # ζ16 = e^{-2πi/16}
    # 샘플 행렬 S[h, l] = ζ16^{out_nibble(h,l)}
    S = np.empty((16, 16), dtype=np.complex128)
    for h in range(16):
        for l in range(16):
            x = (h << 4) | l
            y = gf_mul_const(x, k)  # GF(2^8) 곱
            if which == "hi":
                val = (y >> 4) & 0xF
            elif which == "lo":
                val = y & 0xF
            else:
                raise ValueError("which must be 'hi' or 'lo'")
            S[h, l] = z16 ** val

    # 2D IFFT ⇒ f(h,l) = Σ c[p,q] ζ^{p h + q l}
    C = np.fft.ifft2(S)  # shape (16, 16), 이미 1/256 정규화 포함

    # 희소화 & 리스트로 직렬화
    entries = []
    for p in range(16):
        for q in range(16):
            c = C[p, q]
            if abs(c) > tol:
                entries.append((p, q, float(c.real), float(c.imag)))
    return entries


def save_coeff(mult: int, which: str, entries):
    obj = {"entries": entries, "multiplier": mult, "which": which, "domain": "zeta16", "size": 16}
    out_path = OUT_DIR / f"gf_mult{mult}_{which}_coeffs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), indent=0)
    print(f"saved: {out_path} (nonzeros={len(entries)})")


# ===== (선택) 정확성 검증: 복원 후 위상→니블 매핑 =====
def _phase_to_nibble(z: np.ndarray) -> np.ndarray:
    ang = np.angle(z)
    k = (-ang * 16) / (2 * np.pi)
    return np.mod(np.rint(k), 16).astype(np.uint8)


def quick_verify(mult: int, which: str, entries) -> bool:
    z16 = np.exp(-2j * np.pi / 16)
    # 재합성: f̂(h,l) = Σ c[p,q] ζ^{p h + q l}
    C = np.zeros((16, 16), dtype=np.complex128)
    for (p, q, re, im) in entries:
        C[p, q] = complex(re, im)
    ok = True
    for h in range(16):
        for l in range(16):
            z = 0 + 0j
            for p in range(16):
                for q in range(16):
                    if C[p, q] != 0:
                        z += C[p, q] * (z16 ** (p * h + q * l))
            nib = int(_phase_to_nibble(np.array([z]))[0])
            x = (h << 4) | l
            y = gf_mul_const(x, mult)
            ref = (y >> 4) & 0xF if which == "hi" else (y & 0xF)
            if nib != ref:
                ok = False
                # 첫 실패만 출력
                print(f"[verify fail] mult={mult} {which}: (h,l)=({h},{l}) got={nib} ref={ref}")
                return ok
    return ok


if __name__ == "__main__":
    # 필요한 계수만 생성: 2/3 × (hi/lo)
    for mult in (1, 2, 3, 9, 11, 13, 14):
        for which in ("hi", "lo"):
            entries = two_var_coeffs_for_multiplier(mult, which, tol=1e-12)
            save_coeff(mult, which, entries)
            assert quick_verify(mult, which, entries), f"verification failed for mult={mult} which={which}"
    print("All done & verified.")
