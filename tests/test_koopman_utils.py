import pytest
import torch
from config import M, T as T_DATA

# --- CONSTANTS ---
TOLERANCE_SCAN  = 1e-5
TOLERANCE_EIGEN = 1e-4
T_VALUES        = [1, 10, 50, T_DATA]  # edit T in config.py to update all tests


# ================================================================================
# 1. PARAMETER INITIALIZATION SHAPE TESTS
# ================================================================================
def test_init_params_shapes():
    """nu_log, theta_log must be (M,) and P must be (M, M)."""
    from models.koopman_utils import init_koopman_params

    nu_log, theta_log, P = init_koopman_params(M)

    assert nu_log.shape    == (M,),    f"nu_log shape {nu_log.shape}, expected ({M},)"
    assert theta_log.shape == (M,),    f"theta_log shape {theta_log.shape}, expected ({M},)"
    assert P.shape         == (M, M),  f"P shape {P.shape}, expected ({M}, {M})"


def test_init_params_types():
    """nu_log and theta_log must be real; P must be complex."""
    from models.koopman_utils import init_koopman_params

    nu_log, theta_log, P = init_koopman_params(M)

    assert not nu_log.is_complex(),    "nu_log should be real"
    assert not theta_log.is_complex(), "theta_log should be real"
    assert P.is_complex(),             "P should be complex"


# ================================================================================
# 2. EIGENVALUE STABILITY TESTS
# ================================================================================
def test_lambda_magnitudes_inside_unit_circle():
    """All eigenvalue magnitudes must be strictly in (0, 1)."""
    from models.koopman_utils import init_koopman_params, compute_lambda

    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)

    magnitudes = Lambda.abs()
    assert (magnitudes > 0).all(),   "Eigenvalue magnitudes must be > 0"
    assert (magnitudes < 1).all(),   f"Eigenvalue magnitudes must be < 1, got max {magnitudes.max().item():.6f}"


def test_lambda_shape():
    """Lambda must be a complex vector of shape (M,)."""
    from models.koopman_utils import init_koopman_params, compute_lambda

    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)

    assert Lambda.shape == (M,),   f"Lambda shape {Lambda.shape}, expected ({M},)"
    assert Lambda.is_complex(),    "Lambda must be complex"


def test_lambda_stable_after_perturbation():
    """Eigenvalue magnitudes must remain in (0,1) after a random gradient-like perturbation."""
    from models.koopman_utils import init_koopman_params, compute_lambda

    nu_log, theta_log, _ = init_koopman_params(M)

    # Simulate a gradient update
    nu_log    = nu_log    + 0.1 * torch.randn(M)
    theta_log = theta_log + 0.1 * torch.randn(M)

    Lambda = compute_lambda(nu_log, theta_log)
    magnitudes = Lambda.abs()

    assert (magnitudes > 0).all(), "Magnitudes must be > 0 after perturbation"
    assert (magnitudes < 1).all(), f"Magnitudes must be < 1 after perturbation, got max {magnitudes.max().item():.6f}"


# ================================================================================
# 3. PARALLEL SCAN SHAPE TESTS
# ================================================================================
@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_output_shape(n_steps):
    """parallel_scan output must be (T, M)."""
    from models.koopman_utils import init_koopman_params, compute_lambda, parallel_scan

    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, g0, n_steps)

    assert out.shape == (n_steps, M), f"Expected ({n_steps}, {M}), got {out.shape}"


# ================================================================================
# 4. PARALLEL SCAN MATCHES SEQUENTIAL
# ================================================================================
@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_matches_sequential(n_steps):
    """parallel_scan must match sequential for-loop within tolerance 1e-5."""
    from models.koopman_utils import init_koopman_params, compute_lambda, parallel_scan, sequential_scan

    torch.manual_seed(42)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)

    out_parallel   = parallel_scan(Lambda, g0, n_steps)
    out_sequential = sequential_scan(Lambda, g0, n_steps)

    assert torch.allclose(out_parallel, out_sequential, atol=TOLERANCE_SCAN), \
        f"Max difference: {(out_parallel - out_sequential).abs().max().item()}"


# ================================================================================
# 5. PARALLEL SCAN EDGE CASE T=1
# ================================================================================
def test_parallel_scan_edge_case_T1():
    """parallel_scan must handle T=1 and return shape (1, M)."""
    from models.koopman_utils import init_koopman_params, compute_lambda, parallel_scan

    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, g0, n_steps=1)

    assert out.shape == (1, M), f"Expected (1, {M}), got {out.shape}"


# ================================================================================
# 6. PARALLEL SCAN DETERMINISTIC
# ================================================================================
def test_parallel_scan_deterministic():
    """Same inputs must always produce the same output."""
    from models.koopman_utils import init_koopman_params, compute_lambda, parallel_scan

    torch.manual_seed(42)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)

    out1 = parallel_scan(Lambda, g0, n_steps=20)
    out2 = parallel_scan(Lambda, g0, n_steps=20)

    assert torch.allclose(out1, out2), "parallel_scan is not deterministic"


# ================================================================================
# 7. PARALLEL SCAN NO NaN OR INF
# ================================================================================
@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_no_nan_inf(n_steps):
    """parallel_scan must not produce NaN or Inf values."""
    from models.koopman_utils import init_koopman_params, compute_lambda, parallel_scan

    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, g0, n_steps)

    assert torch.isfinite(out.abs()).all(), \
        f"NaN or Inf found in parallel_scan output at T={n_steps}"


# ================================================================================
# 8. PARALLEL SCAN STABLE DYNAMICS
# ================================================================================
def test_parallel_scan_stable():
    """Latent state norm must not grow unboundedly over long sequences."""
    from models.koopman_utils import init_koopman_params, compute_lambda, parallel_scan

    torch.manual_seed(42)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, g0, n_steps=T_DATA)

    initial_norm = g0.abs().norm().item()
    final_norm   = out[-1].abs().norm().item()

    assert final_norm <= initial_norm * 10, \
        f"Latent state exploded: initial norm={initial_norm:.3f}, final norm={final_norm:.3f}"


# ================================================================================
# 9. P MATRIX IS INVERTIBLE
# ================================================================================
def test_P_is_invertible():
    """P must be invertible (non-zero determinant)."""
    from models.koopman_utils import init_koopman_params

    _, _, P = init_koopman_params(M)
    det = torch.linalg.det(P).abs().item()

    assert det > 1e-6, f"P is not invertible, det={det:.2e}"