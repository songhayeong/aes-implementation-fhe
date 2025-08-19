import json, numpy as np
from pathlib import Path


def round16_angle(theta):
    """각도 theta를 가장 가까운 16방향으로 스냅한 각도 반환"""
    k = int(np.round((theta % (2 * np.pi)) / (2 * np.pi / 16))) % 16
    return 2 * np.pi * k / 16.0


def build_snap_samples(n_samples=4096):
    """원 위에서 샘플 x=e^{i0}, target y=snap(e^(i0))를 생성"""
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    x = np.exp(1j * theta)
    theta_snap = np.array([round16_angle(t) for t in theta])
    y = np.exp(1j * theta_snap)
    return x, y


def fit_poly_ls(x, y, deg=15, ridge=0.0):
    """
    최소제곱으로 P(z)=∑_{k=0..deg} c_k z^k 적합
    - x: 샘플 (원 위)
    - y: 타깃 (스냅된 unit circle 점)
    """
    V = np.vstack([x ** k for k in range(deg + 1)]).T  # Vandermonde
    if ridge > 0:
        # (V^H V + λI) c = V^H y
        VH_V = V.conj().T @ V
        VH_y = V.conj().T @ y
        A = VH_V + ridge * np.eye(deg + 1)
        c = np.linalg.solve(A, VH_y)
    else:
        c, *_ = np.linalg.lstsq(V, y, rcond=None)
    return c  # complex np.array length deg+1


def save_coeffs_json(coeffs, path: Path):
    entries = []
    for k, c in enumerate(coeffs):
        entries.append([int(k), float(c.real), float(c.imag)])
    obj = {"type": "zeta16_snap_1d_poly", "entries": entries}
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # 1) 샘플 만들기
    x, y = build_snap_samples(n_samples=8192)

    # 2) 다항식 적합 (차수 15, 필요시 ridge 조금(1e-6~1e-3) 줘도 좋음)
    coeffs = fit_poly_ls(x, y, deg=15, ridge=1e-5)

    # 3) 저장
    out = Path("zeta16_snap_coeffs.json")
    save_coeffs_json(coeffs, out)
    print(f"saved -> {out.resolve()}")
