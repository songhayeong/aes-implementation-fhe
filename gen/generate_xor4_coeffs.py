"""
Script to generate 2D LUT polynomial coefficients for 4-bit XOR over GF domain.
Outputs JSON Files in generator/coeffs/xor4_coeffs.json
"""
import numpy as np
import json
from pathlib import Path


def fft2_coeffs(vals: np.ndarray) -> np.ndarray:
    """
    Compute 2D polynomial coefficients via inverse 2D FFT:
    a[p, q] = (n^2) * ifft2(vals)[p, q]
    """
    n = vals.shape[0]
    # numpy.fft.ifft2 yields (1/n^2) * sum(vals * exp(2pi * j ...))
    a = np.fft.ifft2(vals) * (n ** 2)
    return a


def make_entries(a: np.ndarray, tol: float = 1e-8) -> list:
    """
     Build list of [i, j, real(a_ij), imag(a_ij)] for non-negligible coeffs.
    """
    entries = []
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            c = a[i, j]
            if abs(c) > tol:
                entries.append([int(i), int(j), float(c.real), float(c.imag)])
    return entries


def main():
    n = 16
    # 16th root of unity
    z = np.exp(-2j * np.pi / n)

    # build table F[p, q] = Zeta^(p XOR q)
    F = np.array([[z ** (p ^ q) for q in range(n)] for p in range(n)], dtype=np.complex128)

    # compute coefficients
    A = fft2_coeffs(F)  # shape (16, 16)
    entries = make_entries(A)

    # write JSON
    out_dir = Path(__file__).parent / "coeffs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "xor4_coeffs.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({"entries": entries}, f, indent=2)

    print(f"Wrote 4-bit XOR LUT coeffs to {out_file}")


if __name__ == '__main__':
    main()
