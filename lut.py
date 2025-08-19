import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from engine_context import EngineContext


def load_coeff1d(path: Path) -> np.ndarray:
    """
    Load 1D LUT polynomial coefficients from JSON.

    JSON format:
    {
        "entries": {
            [k, real, imag],
        }
    }

    Returns a numpy array 'A' of complex128, shape (max_k+1, ), where
    A[k] = real+1j*imag for each entry.
    Missing indices are filled with 0
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    # Determine max index
    max_k = 0
    for entry in data.get('entries', []):
        k = int(entry[0])
        if k > max_k:
            max_k = k
    # Allocate array
    A = np.zeros(max_k + 1, dtype=np.complex128)
    # Fill coefficients
    for entry in data.get('entries', []):
        k, real, imag = entry
        A[int(k)] = complex(real, imag)
    return A


def load_coeff2d(path: Path, size: int) -> np.ndarray:
    """
    Load 2D LUT polynomial coefficients from JSON.

    JSON format:
    {
        "entries" : [
            [i, j, real, imag],
            ...
        ]
    }
    'size' is the dimension of the square LUT. Returns a numpy array
    'A' of shape (size, size) complex128, where A[I, J] = real + 1j * imag.
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    # Allocate matrix
    A = np.zeros((size, size), dtype=np.complex128)
    # Fill coefficients
    for entry in data.get('entries', []):
        i, j, real, imag = entry
        A[int(i), int(j)] = complex(real, imag)
    return A


class LUTEvaluator:
    def __init__(self, ctx: EngineContext, coeffs: Dict[int, Any], domain_size: int):
        self.ctx = ctx
        self.coeffs = coeffs # {k : plaintext_ct_of(a_k)}
        self.domain = domain_size # e.g. 16 for 4-bit LUT, 256 for byte LUT

    def apply(self, ct):
        eng = self.ctx
        # 1) power-basis up to deg = domain // 2
        half = self.domain // 2
        pos = eng.make_power_basis(ct, half)
        basis = {0: eng.add_plain(ct, 1.0)}
        # 2) fill basis[k]
        for k in range(1, self.domain):
            if k <= half:
                basis[k] = pos[k-1]
            else:
                basis[k] = eng.conjugate(pos[self.domain-k-1])
        # 3) Horner-less sum coeffs(k) * basis[k]
        res = eng.multiply(ct, 0.0)
        for k, pt in self.coeffs.items():
            if k == 0:
                res = eng.add(res, pt)
            else:
                res = eng.add(res, eng.multiply(basis[k], pt))
        return res