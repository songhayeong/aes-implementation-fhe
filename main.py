import numpy as np
from pathlib import Path
from engine_context import EngineContext
from pipeline import AESPipeline
from lut import load_coeff1d, load_coeff2d
import time

# -------- setting ----------

# CKKS parameter (paper와 동일하게 세팅 N=2^16, slots=2^15)
CKKS_PARAMS = {
    "signature": 2,
    "max_level": 17,
    "mode": "cpu",
    "thread_count": 4
}

BASE_DIR = Path(__file__).parent / "generator" / "coeffs"
XOR4_PATH = BASE_DIR / "xor4_coeffs.json"
SUB_HI_PATH = BASE_DIR / "mod256_to_16_hi.json"
SUB_LO_PATH = BASE_DIR / "mod256_to_16_lo.json"


# ------ Main ------------
def main():
    # 1) FHE 엔진 초기화
    ctx = EngineContext(**CKKS_PARAMS)

    # 2) LUT 계수 로드
    xor4 = load_coeff2d(XOR4_PATH, size=16)
    sub_hi = load_coeff1d(SUB_HI_PATH)
    sub_lo = load_coeff1d(SUB_LO_PATH)

    coeffs = {
        'xor4': load_coeff2d(XOR4_PATH, size=16),
        'sub_hi': load_coeff1d(SUB_HI_PATH),
        'sub_lo': load_coeff1d(SUB_LO_PATH)
    }

    # 3) AES pipeline 생성
    aes_pipe = AESPipeline(ctx, coeffs)

    # 4) 테스트할 평문 상태 및 라운드 키
    np.random.seed(0)
    state = np.random.randint(0, 256, size=16, dtype=np.uint8)
    key = np.random.randint(0, 256, size=16, dtype=np.uint8)

    print("Plaintext state:", state)
    print("Round key:", key)

    # 5) 평문 단계에서 분할 암호화
    ct_hi, ct_lo = aes_pipe.encoder.encode(state)
    ki_hi, ki_lo = aes_pipe.encoder.encode(key)

    start = time.perf_counter()
    # 6) AddRoundKey
    ark_hi, ark_lo = aes_pipe.ark(ct_hi, ct_lo, ki_hi, ki_lo)
    end = time.perf_counter()

    # 7) SubBytes
    sb_hi, sb_lo = aes_pipe.sub_bytes(ark_hi, ark_lo)

    # 추가 : ShiftRows, MixColumns 라운드 등을 여기에 순차적 호출
    # Ex) sr_hi, sr_lo = aes_pipe.shift_rows(sb_hi, sb_lo)
    #     mc_hi, mc_lo = aes_pipe.mix_columns(sr_hi, sr_lo)

    # 8) 복호화 및 결과 확인
    result = aes_pipe.encoder.decode(ark_hi, ark_lo)
    expected = state ^ key

    print("HE Result:   ", result)
    print("expected:  ", expected)
    print(f"AddRoundKey took {end - start : .6f} seconds")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# 최대한 직관적으로 심플하게 코드를 구현하자. 내가 지금 너어무 어렵게 생각하는거 같은데... 그럴 필요가 없을 것 같단말이지 ?

# 6.1 A Brief Overview

"""
    The AES-128 consists of 10 rounds of the same operations with different round keys.
    Each round is operated on 4 x 4 matrix of bytes, where each bytes are considered as an element of a Galois field GF(2^8).
    We use n = 2^4 as the fact that 4 divides 8 makes the algorithm easier to understand and implement.
    
    Each round function of AES consists of four operations : AddKey, SubBytes, ShiftRows, and MixColumns.
    
    AddKey is simply XOR of the 16-byte round key and current state, as XOR is a bit-wise operation, we can evaluate this
    with simple 4-to-4 LUT. Besides, it is observed that LUT for XOR is sparse, i.e., the LUT polynomial has many zeros in its
    coefficients, and thus we can skip some ciphertext-ciphertext multiplications. 
    
    SubBytes is S-box look up which is finding alpha^-1 for  alpha in GF(2^8), considering each byte in the state as element
    of GF(2^8), considering each byte in the state as element of GF(2^8). This is naturally done by the proposed method using
    8-to-8 LUT evaluation.
    
    In ShiftRows step, we consider the state as 4 x 4 matrix and shift each row by some amount. It can be done by rotation 
    operation and inner product with indicator plaintext vectors, which have only one and zeros as elements.
    
    MixCol multiplies the state 4 x 4 matrix by a fixed 4 x 4 matrix in GF(2^8)^(4*4). The multiplication if GF(2^8) is done
    by an 8-to-8 LUT evaluation, where we find the LUT evaluation polynomial for multiplication by given amount^2.
    As in [GHS12], we merge ShiftRows and MixColumns steps and reduce the computation.
    
    Parameter selection and implementation details.
    - To meet 128-bit security, we use N = 2^16, Q = 2^1658, and scaling factor ~ 2^59. For ease of implementation,  
      we do bootstrapping after each LUT evaluation, and each LUT has depth 5.
      There are 10 rounds in AES-128, and we do noise reduction every round for easy implementation.
      
    
    ### 6.2 Homomorphic Evaluation of Basic Operations.
    
    Representation of AES state. -> We divide a byte into two 4-bit numbers, The left half and the right half. 
    A half bytes is encoded into C = {zeta^j}j = 0 ~ 15, where zeta = exp(-2pi * i / 16) thus each slot has a codeword
    c in C. For ease of understanding and implementation, we use separate ciphertexts for left half bytes and right half bytes.
    
    We choose a CKKS parameter with ring dimension N = 2^16, whose maximum number of slots is 2^15.
    As the state requires 16 bytes, we put 2^15 / 16 = 2048 ciphertext batched in two ciphertexts (each slot in the same
    position represents the left and right half of the same bits). Following Gentry-Halevi-Smart [GHS12], we place the 
    16 bytes of the AES state in plaintext slots using column-first ordering, namely, we have.
    
    a = [a_00, a_10, a_20, a_30, a_01, a_11, a_21, a_31, a_02, a_12, a_22, a_32, a_03, a_13, a_23, a_33]
    
    representing the input plaintext matrix
     
    A = | a00 a01 a02 a03 |
        | a10 a11 a12 a13 |
        | a20 a21 a22 a23 |
        | a30 a31 a32 a33 |
    
    The batching is done by placing elements of a by the gap of 2^15 / 16 = 2048. For example, let B and wvr are other AES
    states we are batching. Then the coefficient slot will look like the following array.
    N = 2^16 -> total length? 
    
    By doing this, we can rotate each state by r independently by applying a rotation of 2048r. Finally, we encode
    each element with encode and encrypt.
    
    AddKey - Here, we assume the AES round key is given as a CKKS ciphertext. AddKey is simply an XOR with AES state and
    encryption key, it can be done by 8-to-4 LUT. In our experiment, we used two ciphertexts, which require two executions
    of 8-to-4 LUTs, and it took 1.63s. Note that the polynomials for XOR have many zeros in them and only a quarter of them are 
    non-zeros. thus, the homomorphic evaluation of the polynomial can be done very fast.
    
    SubBytes - SubBytes and its inverse InvSubBytes can be implemented as 8-to-8 LUT. An 8-to-8 LUT is done by two-8-to-4 LUTs
    As those two LUTs share the inputs, we can calculate the power basis and reuse it. Thus, it took 0.94s in our experiment,
    which is almost half of AddKey. We do the same LUTs on each slot.
    
    ShiftRows and MixColumns - As in [GHS12], we can merge the ShiftRows and MixColumns steps. We explain the AES decryption in
    detail, but the encryption can also be d
    
    Using a CKKS-encrypted AES state instead of conventional plaintext extends the duration of the first AddRoundKey step.
    Combining invSubBytes with AddKey into a LUT presents additional optimization opportunities.
    This list is not exhaustive; other optimization avenues remain unexplored.
    
"""
