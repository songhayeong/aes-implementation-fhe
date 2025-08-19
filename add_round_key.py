import numpy as np
from typing import Tuple, Any, Dict
from engine_context import EngineContext
from utils import ZetaEncoder
import json
from pathlib import Path

from xor4_lut import XOR4LUT


def load_xor4_coeffs(path: Path) -> np.ndarray:
    """
    JSON 형식의 4-bit XOR LUT 계수를
    shape = (16, 16) complex numpy 배열로 반환
    """
    data = json.loads(path.read_text(encoding='utf-8'))
    A = np.zeros((16, 16), dtype=np.complex128)
    for i, j, real, imag in data["entries"]:
        A[i, j] = complex(real, imag)
    return A

XOR4_JSON = Path(__file__).parent / "generator" / "coeffs" / "xor4_coeffs.json"
XOR4_COEFFS = load_xor4_coeffs(XOR4_JSON)


# --- Load 1D LUT coefficients for SubBytes hi/lo ---
def load_lut1d(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding='utf-8'))
    # determine max index
    max_k = max(entry[0] for entry in data['entries'])
    A = np.zeros(max_k+1, dtype=np.complex128)
    for k, real, imag in data['entries']:
        A[int(k)] = complex(real, imag)
    return A

dir_coeffs = Path(__file__).parent / "generator" / "coeffs"
# SUB_HI_JSON = dir_coeffs / "mod256_to_16_hi.json"
# SUB_LO_JSON = dir_coeffs / "mod256_to_16_lo.json"
# SUB_HI_COEFF = load_lut1d(SUB_HI_JSON) # degree-255
# SUB_LO_COEFF = load_lut1d(SUB_LO_JSON)



class AESFHE:
    """
    AES=128 homomorphic pipeline : state split into two 4-bit ciphertexts,
    operations on hi/lo parts via 4-bit XOR LUTs.
    """
    def __init__(self, ctx: EngineContext):
        self.ctx = ctx
        self.sc = ctx.engine.slot_count  # 이때의 slot count는 2^15를 만족해야함 paper 따라가기 그래야 Gap이 2048이 된다.

        # 1) 0이 아닌 계수만 plaintext로 미리 인코딩
        self._xor4_plain: Dict[Tuple[int, int], Any] = {}
        for p in range(16):
            for q in range(16):
                c= XOR4_COEFFS[p, q]
                if abs(c) > 1e-12:
                    vec = np.full(self.sc, c, dtype=np.complex128)
                    self._xor4_plain[(p, q)] = ctx.encode(vec)

    def encode_state(self, state: np.ndarray) -> Tuple[Any, Any]:
        """
        Split 16-byte state into hi/lo 4-bit vectors, encrypt separately.
        Returns (ct_hi, ct_lo).
        """
        assert state.shape == (16, )
        # split
        hi = (state >> 4) & 0x0F
        lo = state & 0x0F
        # encode to zeta16
        z_hi = ZetaEncoder.to_zeta(hi.astype(np.uint8), 16)
        z_lo = ZetaEncoder.to_zeta(lo.astype(np.uint8), 16)
        # pack into slot vectors
        vec_hi = np.ones(self.sc, dtype=np.complex128)
        vec_lo = np.ones(self.sc, dtype=np.complex128)
        stride = self.sc // 16
        for i in range(16):
            vec_hi[i*stride] = z_hi[i]
            vec_lo[i*stride] = z_lo[i]
        return self.ctx.encrypt(vec_hi), self.ctx.encrypt(vec_lo)

    def decrypt_state(self, ct_hi: Any, ct_lo: Any) -> np.ndarray:
        """
        Decrypt hi/lo ciphertexts and recombine into 16-byte state.
        """
        vec_hi = self.ctx.decrypt(ct_hi)
        vec_lo = self.ctx.decrypt(ct_lo)
        stride = self.sc // 16
        out = np.empty(16, dtype=np.uint8)
        for i in range(16):
            hi_val = ZetaEncoder.from_zeta(np.array([vec_hi[i*stride]]), 16)[0] # 이게 띄워진만큼 가져오는건가 싶다
            lo_val = ZetaEncoder.from_zeta(np.array([vec_lo[i*stride]]), 16)[0]
            out[i] = (hi_val << 4) | lo_val
        return out

    def _build_power_basis_16(self, ct: Any) -> Dict[int, Any]:
        """
        Build ct^k for k=0..15 in Zeta16 domain using power basis + conjugate.
        """
        eng = self.ctx
        pos = eng.make_power_basis(ct, 8)
        basis: Dict[int, Any] = {0: eng.add_plain(ct, 1.0)}
        for k in range(1, 16):
            if k <= 8:
                basis[k] = pos[k-1]
            else:
                basis[k] = eng.conjugate(pos[16-k-1])
        return basis

    def _xor4(self, a_ct: Any, b_ct: Any) -> Any:
        """
        4-bit XOR: f(a,b) = a xor b를 bivariate polynomial로 평가
        """
        eng = self.ctx
        A = self._build_power_basis_16(a_ct)
        B = self._build_power_basis_16(b_ct)

        res = eng.multiply(a_ct, 0.0)
        for (p, q), pt_coeff in self._xor4_plain.items():
            term = eng.multiply(A[p], B[q])
            res = eng.add(res, eng.multiply(term, pt_coeff))
        return res

    def add_round_key(self,
                      ct_hi: Any, ct_lo: Any,
                      key_hi: Any, key_lo: Any) -> Tuple[Any, Any]:
        """
        Homomorphic AddRoundKey: XOR state and key nibble-wise.
        Inputs: ct_hi, ct_lo, key_hi, key_lo ciphertexts.
        Returns: (ct_hi_out, ct_lo_out)
        """
        out_hi = self._xor4(ct_hi, key_hi)
        out_lo = self._xor4(ct_lo, key_lo)
        return out_hi, out_lo


class AddRoundKey:
    def __init__(self, xor4: XOR4LUT):
        self.xor4 = xor4

    def __call__(self, ct_hi, ct_lo, key_hi, key_lo) -> Tuple[Any, Any]:
        return ( self.xor4.apply(ct_hi, key_hi),
                 self.xor4.apply(ct_lo, key_lo))




#------------------Basic Test Harness -------------------------
if __name__ == '__main__':
    # Initialize engine
    ctx = EngineContext(signature=2, use_multiparty=False, thread_count=4, max_level=17, device_id=0)
    aes = AESFHE(ctx)

    # Test encode/decode
    np.random.seed(0)
    state = np.random.randint(0, 256, 16, dtype=np.uint8)
    ct_hi, ct_lo = aes.encode_state(state)
    decoded = aes.decrypt_state(ct_hi, ct_lo)
    print("State round-trip match?", np.array_equal(state, decoded))

    # Test AddRoundKey
    key = np.random.randint(0, 256, 16, dtype=np.uint8)
    hi_k = (key >> 4) & 0x0F
    lo_k = key & 0x0F
    ct_hi_k, ct_lo_k = aes.encode_state(key)
    out_hi, out_lo = aes.add_round_key(ct_hi, ct_lo, ct_hi_k, ct_lo_k)
    result = aes.decrypt_state(out_hi, out_lo)
    expected = state ^ key
    print("AddRoundKey match?", np.array_equal(result, expected))
    print("Expected: ", expected)
    print("Result: ", result)
