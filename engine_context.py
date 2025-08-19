from desilofhe import Engine, Ciphertext
import numpy as np
import time


class EngineContext:

    def __init__(self,
                 signature: int,
                 *,
                 max_level: int = 17,
                 use_bootstrap: bool = True,
                 use_multiparty: bool = False,
                 mode: str = 'cpu',
                 device_id: int = 0,
                 thread_count: int):
        if signature == 1:
            self.engine = Engine(
                mode=mode,
                use_bootstrap=use_bootstrap,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id,
            )
        elif signature == 2:
            self.engine = Engine(
                max_level=max_level,
                mode=mode,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id,
            )
        elif signature == 3:
            self.engine = Engine(
                mode=mode,
                use_multiparty=use_multiparty,
                thread_count=thread_count,
                device_id=device_id
            )

        else:
            raise ValueError(f"Unsupported signature: {signature}")

        self.secret_key = self.engine.create_secret_key()
        self.public_key = self.engine.create_public_key(self.secret_key)
        self.relinearization_key = self.engine.create_relinearization_key(self.secret_key)
        self.conjugation_key = self.engine.create_conjugation_key(self.secret_key)
        self.rotation_key = self.engine.create_rotation_key(self.secret_key)

        self.bootstrap_key = self.engine.create_bootstrap_key(self.secret_key)

        # bootstrap state
        self._bs_count = 0
        self._bs_total_s = 0.0

    def encrypt(self, data: np.ndarray):
        return self.engine.encrypt(data, self.public_key)

    def decrypt(self, ct) -> np.ndarray:
        return self.engine.decrypt(ct, self.secret_key)

    def encode(self, vec: np.ndarray):
        return self.engine.encode(vec)

    def multiply(self, a, b):
        if isinstance(a, Ciphertext) and isinstance(b, Ciphertext):
            return self.engine.multiply(a, b, self.relinearization_key)
        return self.engine.multiply(a, b)

    def add(self, a, b):
        return self.engine.add(a, b)

    def sub(self, a, b):
        return self.engine.subtract(a, b)

    def add_plain(self, ct, val):
        """
        Add plaintext (scalar or vector). Complex일 경우 반드시 encode 경유.
        """
        # complex 스칼라/배열은 무조건 encode
        if np.iscomplexobj(val):
            pt = self.engine.encode(
                np.full(self.engine.slot_count, val, dtype=np.complex128)
                if np.isscalar(val) else
                np.asarray(val, dtype=np.complex128)
            )
            return self.engine.add(ct, pt)
        # 실수 스칼라는 엔진의 add_plain 경로 사용
        try:
            return self.engine.add_plain(ct, float(val))
        except Exception:
            # fallback: encode 후 add
            pt = self.engine.encode(
                np.full(self.engine.slot_count, val, dtype=np.complex128)
                if np.isscalar(val) else
                np.asarray(val, dtype=np.complex128)
            )
            return self.engine.add(ct, pt)

    def make_power_basis(self, ct, degree: int):
        return self.engine.make_power_basis(ct, degree, self.relinearization_key)

    def conjugate(self, ct):
        return self.engine.conjugate(ct, self.conjugation_key)

    def multiply_plain(self, ct, val):
        """
        Multiply ciphertext by plaintext scalar or vector.
        Complex일 경우 반드시 encode 경유 (엔진이 실수 스칼라만 안전).
        """
        if np.isscalar(val):
            if np.iscomplexobj(val):
                # complex scalar → encode 후 곱
                pt = self.engine.encode(
                    np.full(self.engine.slot_count, val, dtype=np.complex128)
                )
                return self.engine.multiply(ct, pt)
            # 실수 스칼라는 엔진 곱 경로
            return self.engine.multiply(ct, float(val))

        # 벡터일 때는 항상 encode
        arr = np.asarray(val)
        dtype = np.complex128 if np.iscomplexobj(arr) else np.float64
        pt = self.engine.encode(arr.astype(dtype, copy=False))
        return self.engine.multiply(ct, pt)

    def rotate(self, ct, steps: int):
        """
        Shift (slot per operation) operation
        기본적으로 self.rotation_key 사용
        """
        return self.engine.rotate(ct, self.rotation_key, steps)

    def relinearize(self, ct):
        """
        Relinearize only degree-2 ciphertexts (3-polynomials);
        leave degree-1 ciphertexts unchanged.
        """
        try:
            return self.engine.relinearize(ct, self.relinearization_key)
        except RuntimeError as e:
            # If it's not a degree-2 ciphertext, skip relinearization
            if "should have 3 polynomials" in str(e):
                return ct
            raise

    def bootstrap(self, ct):
        """
        Refresh ciphertext modulus level using bootstrapping
        time / call counter recording
        """
        t0 = time.perf_counter()
        out = self.engine.bootstrap(
            ct,
            self.relinearization_key,
            self.conjugation_key,
            self.bootstrap_key
        )
        dt = time.perf_counter() - t0
        self._bs_count += 1
        self._bs_total_s += dt
        return out

    # useful method for bootstrap
    def bootstrap_stats(self):
        avg = (self._bs_total_s / self._bs_count) if self._bs_count else 0.0
        return {"count": self._bs_count, "total_s": self._bs_total_s, "avg_s": avg}

    def reset_bootstrap_stats(self):
        self._bs_total_s = 0
        self._bs_count = 0

    def to_ntt(self, x):
        return self.engine.ntt(x)

    def to_intt(self, x):
        return self.engine.intt(x)

    # make_power_basis가 non-NTT/ level 요구할 때 자동 대처
    def make_power_basis_safe(self, ct, deg):
        try:
            # 바로 시도
            return self.engine.make_power_basis(ct, deg, self.relinearization_key)
        except RuntimeError as e:
            msg = str(e)
            # NTT 상태라서 실패 -> INTT 후 재시도
            if "NTT" in msg:
                ct = self.to_intt(ct)
                return self.engine.make_power_basis(ct, deg, self.relinearization_key)
            # 레벨 부족 -> (INTT 보정 후) 부트스트랩 -> 재시도
            if "level" in msg or "positive" in msg:
                ct = self.to_intt(ct)
                ct = self.bootstrap(ct)
                return self.engine.make_power_basis(ct, deg, self.relinearization_key)
            raise

    # bootstrap safe version
    def bootstrap_safe(self, ct):
        return self.engine.bootstrap(
            self.to_intt(ct),
            self.relinearization_key,
            self.conjugation_key,
            self.bootstrap_key
        )
