# AES implementation via Homomorphic Encryption
### In DGIST PACL

---

This project implement end-to-end AES-128 encryption and decryption over CKKS using a Zeta16 nibble encoding.
We follow column-first packing.

Each round is built from LUT polynomials (for S-Box/XOR/GF multipliers) plus carefully managed rotations and stability.

1. Fully modular: every AES step is a self-contained modulus with a clear interface.
2. Auditable: per-step debug snapshots (ciphertexts + decoded palins)
3. Correctness: full encrypt -> decrypt round-trip matches the plaintext.

---

### Highligts

- **Zeta16 nibble domain**: Split each byte into (hi, lo) 4-bit nibbles, each encoded on the 16-point codebook (Zeta16 powers).(Conjugate 추가하기 !!!!!). Pack 16 bytes per state with stride slots / 16.
- **XOR via sparse bivariate LUT**: 4-bit XOR polynomial with many zero coefficients -> fast AddRoundKey.
- **S-Box via two 1D LUTs**: Hi/lo polynomials share the same power basis to save work.
- **ShiftRows** via masked per-row rotations under column-first packing.
- **MixColumns (forward)** as GF(2)*state xor GF(3)*rot1 xor rot2 xor rot3, where rotk is column up-shift by k implemented as a single packed rotate. (diagonal 텀에 대해 행렬곱을 했을시 이 믹스컬럼결과가 똑같이 나온다는것에 대한 것에서 아이디어를 얻음)
- **InvMixColumns** via bivariate LUTs for GF(2^8) multipliers x (9, 11, 13, 14), then XOR accumulation.

---

### Repository layout (modules & responsibilities)
```
generator/coeffs/
    xor4_coeffs.json                                    # 4-bit XOR (16x16)
    mod256_to_16_hi.json                                # SubBytes(hi) 1D LUT
    mod256_to_16_lo.json                                # SubBytes(lo) 1D LUT
    gf_mult(2,3,9,11,13,14)_{hi,lo}_coeffs.json         # GF multipliiers for mix / invmix
    
    
engine_context                                          # Thin wrapper over CKKS engine (rotate, multiply, bootstrap,
                                                        # make_power_basis, encode/decode helpers, domain conversions)
                                               
state_encoder.py                                        # Zeta16 nibble split/pack/unpack for 16-byte states

lut.py                                                  # JSON loaders (1D/2D) -> sparse numpy arrays
xor4_lut.py                                             # XOR4LUT: 4-bit XOR via Σ c[p,q] X^p Y^q (bivariate LUT)
sub_bytes_lut.py                                        # SubBytes / InvSubBytes via two 1D polynomials (hi/lo)

shift_rows.py                                           # ShiftRows (masked per-row roates in column-first layout)
inv_shiftrows.py                                        # Inverse ShiftRows (mirros shift_rows)

mixcol_final.py                                         # Forward MixColumns
                                                        # - gf_mult_2/gf_mult_3 LUT eval + column-shifts
                                                        # - XOR accumulation with optional post-XOR snap

invmixcolumns_fhe.py                                    # Inverse MixColumns
                                                        # -gf_mult_9/11/13/14 via bivariate LUTs
                                                        # - rotate, LUT, XOR, optional snaps
                                                        
pipeline.py                                             # AESPipeline: orchestration for full AES-128
                                                        # - add_round_key / sub_bytes / (inv_)shiftrows / (inv_)mixcolumns
                                                        # - (optional) renorm between steps
                                                        # - rich debug logging per stage
                                                        
test_aes_pipeline_roundtrip.py                          # Full encrypt -> decrypt round-trip with timing

```

### How the modules compose (round flow)

**Encryption**

```angular2html
AESPipeline.encrypt : 
 (1) AddRoundKey (xor4_lut)
  repeat r = 1 .. 9
    (2) SubBytes (sub_bytes_lut)
    (3) ShiftRows (Shift_rows)
    (4) MixColumns (mixcol_final)
    (5) AddRoundKey (xor4_lut)
 (6) Final : SubBytes -> ShiftRows -> AddRoundKey
```

**Decryption**
```angular2html
AESPipeline.decrypt:
 (1) AddRoundKey with rk[10]
  repeat r=9..1:
    (2) InvShiftRows (inv_shiftrows)
    (3) InvSubBytes (sub_bytes_lut with inverse coeffs)
    (4) AddRoundKey
    (5) InvMixColumns (invmixcolumns_fhe)
 (6) Final : InvShiftRows -> InvSubBytes -> AddRoundKey with rk[0]
```

**Debugging hooks**
- Ecah call records a snapshot into a debug dict : ciphertexts + decoded plains (pipepline._log_pair)
- You can diff any stage against a NumPy plaintext reference.
---

### Packing & rotations
- **Column-first packing** : state laid out as
    [a00, a10, a20, a30, a01, a11, a21, a31, a02, a12, a22, a32, a03, a13, a23, a33]
- **Stride** : stride = slot_count / 16
- **Row masks** : plaintext masks keep only row r at indices (r+4c) * stride.
   Rotate masked parts by ±k*stride and sum to implement row shifts within columns.
- Column up-shift by k (row-major intuition): implement as **one rotate** -4*k*stride on the packed vector (fast path):
  This yields the correct per-column cyclic shift without per-row masking.

---

### Performance 
| Numbers depend on CKKS params, bootstraps, and whether stability "snap" is enabled.

- Encryption (10 rounds) : 8,208s (≈ 810~830s per round)
- Decryption (10 rounds) : 13,938s (≈ 1,495~1,527s per round)
- Total (enc + dec) : 22,146s
- Round-trip: recovered plaintext == original (True)

---

### Implementation notes
- **Domain discipline** : We ensure power-basis builds and LUT evals are done on the expected domain.
 When in doubt, we "snap" (decode -> re-encode) to re-anchor to the Zeta16 codebook.
- **Rotate -> XOR drift** : Repeated XOR LUTs after rotations can drift numerically. 
- **Sparse coefficient cache** : For XOR, only non-zero LUT coefficients are encoded as plaintexts-significant speed savings.

---

### Extending / Development plan

1. **Fuse LUTs**
 - SubBytes xor AddRoundKey as a single bivariate nibble LUT (or two 1D + corrector), removing one XOR per byte.
 - Likewise on decryption with InvSubBytes xor AddRoundKey

2. **GHS12 merge refinements**
 - Tighter ShiftRows + MixColumns fusion to reduce rotates/mults further.

3. **Boostrapping optimization**
 - In this simulation, handle bootstrapping only use try-exception. So i think it has more optimization chance.

4. **Parameter optimization**
 - In this simulation, only uses same parameter. 

---

### FAQ
- **Why split bytes into nibbles ?** 

  Makes 8-bit LUTs into 4-bit (or pair of 4-bit) polynomials; cheaper power bases and sparser XOR

- **Why column-first packing?** 

  It aligns with GHS12 rotations and lets us implement ShiftRows/MixColumns as structed rotates + masks.

## Reference

- IACR ePrint **2024/274**. *Main reference for our CKKS/Zeta₁₆ LUT approach (nibble-split AES, XOR and S-Box via multivariate polynomials, ShiftRows/MixColumns packing, and bootstrapping strategy).*  
  https://eprint.iacr.org/2024/274