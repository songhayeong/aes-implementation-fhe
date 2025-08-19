# aes_pipeline.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np

from engine_context import EngineContext
from state_encoder import StateEncoder
from xor4_lut import XOR4LUT
from sub_bytes_lut import SubBytesLUT
from add_round_key import AddRoundKey
from shift_rows import ShiftRows
from inv_shiftrows import InvShiftRows
from mixcol_final import MixColFinal  # forward
from invmixcolumns_fhe import InvMixColumnsFHE  # inverse


class AESPipeline:
    """
    AES-128 encryption/decryption pipeline on CKKS/Zeta16 (hi/lo nibble split).
    coeffs 필요 키:
      - 'xor4'        : (16x16) XOR LUT (complex)
      - 'sub_hi'      : SubBytes(hi) 1D LUT coeff
      - 'sub_lo'      : SubBytes(lo) 1D LUT coeff
      - 'inv_sub_hi'  : InvSubBytes(hi) 1D LUT coeff
      - 'inv_sub_lo'  : InvSubBytes(lo) 1D LUT coeff
    """

    def __init__(
            self,
            ctx: EngineContext,
            coeffs: Dict[str, Any],
            *,
            mixcolumns: MixColFinal | None = None,
            inv_mixcolumns: InvMixColumnsFHE | None = None,
            use_hard_renorm_between_steps: bool = False
    ):
        self.ctx = ctx
        self.encoder = StateEncoder(ctx)
        self.sc = ctx.engine.slot_count
        self.stride = self.sc // 16

        # LUTs
        self.xor4 = XOR4LUT(ctx, coeffs['xor4'])
        self.sub = SubBytesLUT(ctx, coeffs['sub_hi'], coeffs['sub_lo'])
        self.isub = SubBytesLUT(ctx, coeffs['inv_sub_hi'], coeffs['inv_sub_lo'])

        # Linear ops
        self.shift = ShiftRows(ctx)
        self.invshift = InvShiftRows(ctx)

        # MixColumns (forward / inverse)
        self.mix = mixcolumns if mixcolumns is not None else MixColFinal(ctx, self.xor4)
        self.invmix = inv_mixcolumns if inv_mixcolumns is not None else InvMixColumnsFHE(ctx, self.xor4)

        # XOR wrapper
        self.ark = AddRoundKey(self.xor4)

        # stability
        self.use_hard_renorm_between_steps = use_hard_renorm_between_steps

        # round key cache
        self._rk_cache: List[Tuple[Any, Any]] | None = None

    # ---------- utils ----------
    def _renorm_pair(self, hi: Any, lo: Any) -> Tuple[Any, Any]:
        if not self.use_hard_renorm_between_steps:
            return hi, lo
        st = self.encoder.decode(hi, lo)
        return self.encoder.encode(st)

    def _encode_key(self, key_bytes: np.ndarray) -> Tuple[Any, Any]:
        assert key_bytes.shape == (16,)
        return self.encoder.encode(key_bytes.astype(np.uint8))

    def _prepare_round_keys(self, round_keys: List[np.ndarray]) -> List[Tuple[Any, Any]]:
        if self._rk_cache is not None and len(self._rk_cache) == len(round_keys):
            return self._rk_cache
        rk_enc: List[Tuple[Any, Any]] = []
        for rk in round_keys:
            rk_enc.append(self._encode_key(np.asarray(rk, dtype=np.uint8)))
        self._rk_cache = rk_enc
        return rk_enc

    def _log_pair(self, dbg: Dict[str, Any] | None, tag: str, ct_hi: Any, ct_lo: Any, **meta) -> None:
        """
        디버그용 스냅샷 저장
        - cipher : 암호문 쌍
        - plain : 디코드 결과 (실패할 시 None, plain_err에 에러 메시지 기록)
        """
        if dbg is None:
            return
        entry = {"ct_hi": ct_hi, "ct_lo": ct_lo, "meta": meta}
        try:
            entry["plain"] = self.encoder.decode(ct_hi, ct_lo)
        except Exception as e:
            entry["plain"] = None
            entry["plain_err"] = repr(e)
        dbg[tag] = entry

    # ---------- primitive ops ----------
    def add_round_key(self, ct_hi: Any, ct_lo: Any, key_hi: Any, key_lo: Any) -> Tuple[Any, Any]:
        return self.ark(ct_hi, ct_lo, key_hi, key_lo)

    def sub_bytes(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return self.sub.apply(ct_hi, ct_lo)

    def inv_sub_bytes(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return self.isub.apply(ct_hi, ct_lo)

    def shift_rows(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return self.shift.apply(ct_hi, ct_lo)

    def inv_shift_rows(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return self.invshift.apply(ct_hi, ct_lo)

    def mix_columns(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return self.mix(ct_hi, ct_lo)

    def inv_mix_columns(self, ct_hi: Any, ct_lo: Any) -> Tuple[Any, Any]:
        return self.invmix(ct_hi, ct_lo)

    # ---------- encrypt ----------
    def encrypt(self, state: np.ndarray, round_keys: List[np.ndarray],
                debug: Dict[str, Any] | None = None) -> Tuple[Any, Any]:

        if debug is not None: debug.clear()

        ct_hi, ct_lo = self.encoder.encode(np.asarray(state, dtype=np.uint8))

        self._log_pair(debug, "enc.input", ct_hi, ct_lo)

        rk = self._prepare_round_keys(round_keys)

        # Round 0
        k0_hi, k0_lo = rk[0]
        ct_hi, ct_lo = self.add_round_key(ct_hi, ct_lo, k0_hi, k0_lo)
        self._log_pair(debug, "enc.r0.ark", ct_hi, ct_lo)
        ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        self._log_pair(debug, "enc.r0.renorm", ct_hi, ct_lo)

        # Rounds 1..9
        for r in range(1, 10):
            ct_hi, ct_lo = self.sub_bytes(ct_hi, ct_lo)
            ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)

            ct_hi, ct_lo = self.shift_rows(ct_hi, ct_lo)
            ct_hi, ct_lo = self.mix_columns(ct_hi, ct_lo)

            kr_hi, kr_lo = rk[r]
            ct_hi, ct_lo = self.add_round_key(ct_hi, ct_lo, kr_hi, kr_lo)
            ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        # 한라운드만 테스트 !

        # Round 1

        # ct_hi, ct_lo = self.sub_bytes(ct_hi, ct_lo)
        # self._log_pair(debug, "enc.r1.sub", ct_hi, ct_lo)
        # ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        # self._log_pair(debug, "enc.r1.sub.renorm", ct_hi, ct_lo)
        #
        # ct_hi, ct_lo = self.shift_rows(ct_hi, ct_lo)
        # self._log_pair(debug, "enc.r1.sr", ct_hi, ct_lo)
        #
        # ct_hi, ct_lo = self.mix_columns(ct_hi, ct_lo)
        # self._log_pair(debug, "enc.r1.mc", ct_hi, ct_lo)
        #
        # kr_hi, kr_lo = rk[1]
        # ct_hi, ct_lo = self.add_round_key(ct_hi, ct_lo, kr_hi, kr_lo)
        # self._log_pair(debug, "enc.r1.ark", ct_hi, ct_lo)
        # ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        # self._log_pair(debug, "enc.final.sub.renorm", ct_hi, ct_lo)

        # Final (10) Round 10 : SubBytes -> ShiftRows -> AddRoundKey k10
        ct_hi, ct_lo = self.sub_bytes(ct_hi, ct_lo)
        self._log_pair(debug, "enc.final.sub", ct_hi, ct_lo)
        ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        self._log_pair(debug, "enc.final.sub.renorm", ct_hi, ct_lo)

        ct_hi, ct_lo = self.shift_rows(ct_hi, ct_lo)
        self._log_pair(debug, "enc.final.sr", ct_hi, ct_lo)

        k10_hi, k10_lo = rk[10]
        ct_hi, ct_lo = self.add_round_key(ct_hi, ct_lo, k10_hi, k10_lo)
        self._log_pair(debug, "enc.final.ark10", ct_hi, ct_lo)

        ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        self._log_pair(debug, "enc.output", ct_hi, ct_lo)
        return ct_hi, ct_lo

    # encryption decryption debug 과정 필요

    # ---------- decrypt ----------
    def decrypt(self, ct_hi: Any, ct_lo: Any, round_keys: List[np.ndarray],
                debug: Dict[str, Any] | None = None) -> Tuple[Any, Any]:
        """
        표준 AES 역연산 순서(한 라운드만 테스트하는 버전도 함께 태깅)
        """
        if debug is not None: debug.clear()

        rk = self._prepare_round_keys(round_keys)
        self._log_pair(debug, "dec.input", ct_hi, ct_lo)

        # 초기 AddRoundKey with rk[10]
        kn_hi, kn_lo = rk[10]
        ct_hi, ct_lo = self.add_round_key(ct_hi, ct_lo, kn_hi, kn_lo)
        self._log_pair(debug, "dec.init.ark10", ct_hi, ct_lo)
        ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        self._log_pair(debug, "dec.init.ark10.renorm", ct_hi, ct_lo)

        # # ----- 한 라운드만 테스트 (Round 9) -----
        # ct_hi, ct_lo = self.inv_shift_rows(ct_hi, ct_lo)
        # self._log_pair(debug, "dec.r9.isr", ct_hi, ct_lo)
        #
        # ct_hi, ct_lo = self.inv_sub_bytes(ct_hi, ct_lo)
        # self._log_pair(debug, "dec.r9.isb", ct_hi, ct_lo)
        # ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        # self._log_pair(debug, "dec.r9.isb.renorm", ct_hi, ct_lo)
        #
        # kr_hi, kr_lo = rk[1]
        # ct_hi, ct_lo = self.add_round_key(ct_hi, ct_lo, kr_hi, kr_lo)
        # self._log_pair(debug, "dec.r9.ark1", ct_hi, ct_lo)
        # ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        # self._log_pair(debug, "dec.r9.ark1.renorm", ct_hi, ct_lo)
        #
        # ct_hi, ct_lo = self.inv_mix_columns(ct_hi, ct_lo)
        # self._log_pair(debug, "dec.r9.imc", ct_hi, ct_lo)
        # ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        # self._log_pair(debug, "dec.r9.imc.renorm", ct_hi, ct_lo)

        for r in range(9, 0, -1):
            ct_hi, ct_lo = self.inv_shift_rows(ct_hi, ct_lo)
            ct_hi, ct_lo = self.inv_sub_bytes(ct_hi, ct_lo)
            ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)

            kr_hi, kr_lo = rk[r]
            ct_hi, ct_lo = self.add_round_key(ct_hi, ct_lo, kr_hi, kr_lo)
            ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)

        # 마지막 라운드 (r=0): InvShiftRows → InvSubBytes → AddRoundKey with rk[0]
        ct_hi, ct_lo = self.inv_shift_rows(ct_hi, ct_lo)
        self._log_pair(debug, "dec.final.isr", ct_hi, ct_lo)

        ct_hi, ct_lo = self.inv_sub_bytes(ct_hi, ct_lo)
        self._log_pair(debug, "dec.final.isb", ct_hi, ct_lo)
        ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        self._log_pair(debug, "dec.final.isb.renorm", ct_hi, ct_lo)

        k0_hi, k0_lo = rk[0]
        ct_hi, ct_lo = self.add_round_key(ct_hi, ct_lo, k0_hi, k0_lo)
        self._log_pair(debug, "dec.final.ark0", ct_hi, ct_lo)
        ct_hi, ct_lo = self._renorm_pair(ct_hi, ct_lo)
        self._log_pair(debug, "dec.output", ct_hi, ct_lo)

        return ct_hi, ct_lo
