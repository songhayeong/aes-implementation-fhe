import numpy as np


class ZetaEncoder:
    """
    Encode/decode integers via roots of unity for CKKS slots.
    """
    @staticmethod
    def to_zeta(arr: np.ndarray, modulus: int) -> np.ndarray:
        # arr % modulus ensures in domain
        zeta = np.exp(-2j * np.pi / modulus)
        return zeta ** (arr % modulus)

    @staticmethod
    def from_zeta(z_arr: np.ndarray, modulus: int) -> np.ndarray:
        # recover integer by angle
        angle = np.angle(z_arr)
        k = (-angle * modulus) / (2 * np.pi)
        return np.mod(np.rint(k), modulus).astype(np.uint8)