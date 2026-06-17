"""
Tests for models/koopman_utils.py

Post-refactor note: init_koopman_params' third return is now P_inv, the input
map into eigenspace, learned directly. It is never inverted. The old
"P must be invertible" contract is gone; full-rank is now a nice-to-have
(a degenerate input map would be a poor init), not a correctness requirement.
"""

import math
import time
import pytest
import torch
import numpy as np

from models.koopman_utils import (
    init_koopman_params,
    compute_lambda,
    parallel_scan,
    sequential_scan,
    compute_pearson,
)
from config import M, T as T_DATA, R_MIN, R_MAX, N_ROIS

TOLERANCE_SCAN = 1e-4
TOLERANCE_EIGEN = 1e-4
T_VALUES = [1, 2, 3, 4, 7, 8, 9, 16, 17, 50, T_DATA]


# ================================================================================
# 1. init_koopman_params
# ================================================================================

def test_init_params_shapes():
    """nu_log and theta_log must be (M,); P_inv must be (M, M)."""
    nu_log, theta_log, P_inv = init_koopman_params(M)
    assert nu_log.shape == (M,), f"nu_log: expected ({M},), got {nu_log.shape}"
    assert theta_log.shape == (M,), f"theta_log: expected ({M},), got {theta_log.shape}"
    assert P_inv.shape == (M, M), f"P_inv: expected ({M}, {M}), got {P_inv.shape}"


def test_init_params_types():
    """nu_log and theta_log must be real; P_inv must be complex."""
    nu_log, theta_log, P_inv = init_koopman_params(M)
    assert not nu_log.is_complex(), "nu_log should be real"
    assert not theta_log.is_complex(), "theta_log should be real"
    assert P_inv.is_complex(), "P_inv should be complex"


def test_P_inv_is_full_rank():
    """
    P_inv should be full rank at init. Post-refactor this is a quality check on
    the initialization (a rank-deficient input map would collapse eigenspace
    directions), NOT a correctness requirement — P_inv is used directly and
    never inverted, so a rank deficiency would not crash, only degrade.
    """
    _, _, P_inv = init_koopman_params(M)
    rank = torch.linalg.matrix_rank(P_inv).item()
    assert rank == M, f"P_inv is not full rank: rank={rank}, expected {M}"


def test_init_magnitudes_within_ring():
    """At init, |Lambda| must lie in [r_min, r_max]."""
    torch.manual_seed(0)
    mags = torch.cat([
        compute_lambda(*init_koopman_params(M)[:2]).abs() for _ in range(50)
    ])
    assert (mags >= R_MIN - TOLERANCE_EIGEN).all(), \
        f"min |Lambda| {mags.min().item():.4f} < r_min {R_MIN}"
    assert (mags <= R_MAX + TOLERANCE_EIGEN).all(), \
        f"max |Lambda| {mags.max().item():.4f} > r_max {R_MAX}"


def test_lambda_magnitude_median_above_midpoint():
    """Median |Lambda| above midpoint confirms ring-uniform (long-memory bias)."""
    torch.manual_seed(42)
    all_mags = []
    for _ in range(100):
        nu_log, theta_log, _ = init_koopman_params(M)
        all_mags.append(compute_lambda(nu_log, theta_log).abs())
    magnitudes = torch.cat(all_mags)
    median = magnitudes.median().item()
    midpoint = (R_MIN + R_MAX) / 2
    assert median > midpoint, \
        f"Median |Lambda| {median:.4f} below midpoint {midpoint:.4f} — short-memory bias"


# ================================================================================
# 2. compute_lambda
# ================================================================================

def test_lambda_shape():
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    assert Lambda.shape == (M,), f"Expected ({M},), got {Lambda.shape}"
    assert Lambda.is_complex(), "Lambda must be complex"


def test_lambda_magnitudes_inside_unit_circle():
    nu_log, theta_log, _ = init_koopman_params(M)
    mags = compute_lambda(nu_log, theta_log).abs()
    assert (mags > 0).all(), "Eigenvalue magnitudes must be > 0"
    assert (mags < 1).all(), f"Must be < 1, got max {mags.max().item():.6f}"


def test_lambda_stable_after_perturbation():
    nu_log, theta_log, _ = init_koopman_params(M)
    nu_log = nu_log + 0.1 * torch.randn(M)
    theta_log = theta_log + 0.1 * torch.randn(M)
    mags = compute_lambda(nu_log, theta_log).abs()
    assert (mags > 0).all(), "Magnitudes must be > 0 after perturbation"
    assert (mags < 1).all(), f"Must be < 1 after perturbation, got max {mags.max().item():.6f}"


def test_compute_lambda_exact_formula():
    nu_log = torch.tensor([-0.5, 0.0, 0.3])
    theta_log = torch.tensor([0.1, -0.2, 0.4])
    got = compute_lambda(nu_log, theta_log)
    expected = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
    assert torch.allclose(got, expected, atol=1e-6), "compute_lambda formula mismatch"

    nu_log_anchor = torch.tensor([math.log(-math.log(0.9))])
    theta_log_anchor = torch.tensor([math.log(math.pi / 2)])
    lam = compute_lambda(nu_log_anchor, theta_log_anchor)[0]
    assert lam.imag.item() > 0, "phase pi/2 should give POSITIVE imaginary part"
    assert abs(lam.real.item()) < 1e-5, "phase pi/2 should give ~zero real part"


# ================================================================================
# 3. parallel_scan / sequential_scan
# ================================================================================

def make_u_bar(g0: torch.Tensor, T: int) -> torch.Tensor:
    u_bar = torch.zeros(T, g0.shape[0], dtype=g0.dtype)
    u_bar[0] = g0
    return u_bar


@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_output_shape(n_steps):
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, make_u_bar(g0, n_steps))
    assert out.shape == (n_steps, M), f"Expected ({n_steps}, {M}), got {out.shape}"


@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_matches_sequential(n_steps):
    torch.manual_seed(42)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    u_bar = make_u_bar(g0, n_steps)
    assert torch.allclose(parallel_scan(Lambda, u_bar), sequential_scan(Lambda, u_bar),
                          atol=TOLERANCE_SCAN)


@pytest.mark.parametrize("n_steps", T_VALUES)
def test_scans_match_closed_form(n_steps):
    torch.manual_seed(0)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    u_bar = make_u_bar(g0, n_steps)
    t = torch.arange(0, n_steps).unsqueeze(1)
    expected = (Lambda.unsqueeze(0) ** t) * g0
    assert torch.allclose(parallel_scan(Lambda, u_bar), expected, atol=1e-4)
    assert torch.allclose(sequential_scan(Lambda, u_bar), expected, atol=1e-4)


def test_parallel_scan_edge_case_T1():
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, make_u_bar(g0, 1))
    assert out.shape == (1, M), f"Expected (1, {M}), got {out.shape}"


def test_parallel_scan_deterministic():
    torch.manual_seed(42)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    u_bar = make_u_bar(g0, 20)
    assert torch.allclose(parallel_scan(Lambda, u_bar), parallel_scan(Lambda, u_bar))


@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_no_nan_inf(n_steps):
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, make_u_bar(g0, n_steps))
    assert torch.isfinite(out.abs()).all(), f"NaN/Inf at T={n_steps}"


def test_scan_hand_worked_real():
    Lambda = torch.tensor([2.0, 3.0], dtype=torch.complex64)
    g0 = torch.tensor([1.0, 1.0], dtype=torch.complex64)
    u_bar = make_u_bar(g0, 3)
    expected = torch.tensor([[1, 1], [2, 3], [4, 9]], dtype=torch.complex64)
    assert torch.allclose(parallel_scan(Lambda, u_bar), expected)
    assert torch.allclose(sequential_scan(Lambda, u_bar), expected)


def test_scan_hand_worked_complex():
    Lambda = torch.tensor([1j, 0.5 + 0j], dtype=torch.complex64)
    g0 = torch.tensor([1.0 + 0j, 2.0 + 0j], dtype=torch.complex64)
    u_bar = make_u_bar(g0, 3)
    expected = torch.tensor([[1 + 0j, 2 + 0j], [1j, 1 + 0j], [-1 + 0j, 0.5 + 0j]],
                            dtype=torch.complex64)
    assert torch.allclose(parallel_scan(Lambda, u_bar), expected, atol=1e-6)
    assert torch.allclose(sequential_scan(Lambda, u_bar), expected, atol=1e-6)


def test_scan_with_control_inputs():
    Lambda = torch.tensor([0.5 + 0j], dtype=torch.complex64)
    u_bar = torch.tensor([[1 + 0j], [2 + 0j], [3 + 0j]], dtype=torch.complex64)
    expected = torch.tensor([[1 + 0j], [2.5 + 0j], [4.25 + 0j]], dtype=torch.complex64)
    assert torch.allclose(parallel_scan(Lambda, u_bar), expected, atol=1e-6)
    assert torch.allclose(sequential_scan(Lambda, u_bar), expected, atol=1e-6)


def test_gradients_flow_through_scan():
    torch.manual_seed(0)
    nu_log = torch.randn(M, requires_grad=True)
    theta_log = torch.randn(M, requires_grad=True)
    g0 = torch.randn(M, dtype=torch.complex64)
    Lambda = compute_lambda(nu_log, theta_log)
    out = parallel_scan(Lambda, make_u_bar(g0, 10))
    out.abs().sum().backward()
    for name, p in [("nu_log", nu_log), ("theta_log", theta_log)]:
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is all zero"


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="parallel_scan only outperforms the loop on GPU")
def test_parallel_faster_than_sequential_gpu():
    dev = "cuda"
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log).to(dev)
    g0 = torch.randn(M, dtype=torch.complex64, device=dev)
    T = max(4096, T_DATA)
    u_bar = make_u_bar(g0, T).to(dev)

    def best(fn, reps=5):
        fn(); torch.cuda.synchronize()
        times = []
        for _ in range(reps):
            torch.cuda.synchronize()
            s = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - s)
        return min(times)

    t_par = best(lambda: parallel_scan(Lambda, u_bar))
    t_seq = best(lambda: sequential_scan(Lambda, u_bar))
    assert t_par < 0.5 * t_seq


# ================================================================================
# 4. compute_pearson
# ================================================================================

def test_pearson_output_shape():
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert fc.shape == (N_ROIS, N_ROIS)


def test_pearson_diagonal_is_one():
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert torch.allclose(torch.diag(fc), torch.ones(N_ROIS), atol=1e-5)


def test_pearson_symmetric():
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert torch.allclose(fc, fc.T, atol=1e-5)


def test_pearson_values_in_range():
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert (fc >= -1.0 - 1e-5).all() and (fc <= 1.0 + 1e-5).all()


def test_pearson_no_nan_inf():
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert torch.isfinite(fc).all()


def test_pearson_permutation_equivariant():
    torch.manual_seed(42)
    x = torch.randn(T_DATA, N_ROIS)
    perm = torch.randperm(N_ROIS)
    assert torch.allclose(compute_pearson(x)[perm][:, perm],
                          compute_pearson(x[:, perm]), atol=1e-5)


def test_pearson_constant_roi():
    x = torch.randn(T_DATA, N_ROIS)
    x[:, 0] = 0.0
    assert torch.isfinite(compute_pearson(x)).all()


def test_pearson_deterministic():
    torch.manual_seed(42)
    x = torch.randn(T_DATA, N_ROIS)
    assert torch.allclose(compute_pearson(x), compute_pearson(x))


def test_pearson_perfect_correlation():
    x = torch.randn(T_DATA, N_ROIS)
    x[:, 1] = x[:, 0]
    assert abs(compute_pearson(x)[0, 1].item() - 1.0) < 1e-5


def test_pearson_perfect_anticorrelation():
    x = torch.randn(T_DATA, N_ROIS)
    x[:, 1] = -x[:, 0]
    assert abs(compute_pearson(x)[0, 1].item() + 1.0) < 1e-5


def test_pearson_matches_numpy_oracle():
    torch.manual_seed(42)
    x = torch.randn(T_DATA, N_ROIS)
    assert np.allclose(compute_pearson(x).numpy(),
                       np.corrcoef(x.numpy(), rowvar=False), atol=1e-5)


def test_pearson_hand_worked():
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    assert torch.allclose(compute_pearson(x), torch.ones(2, 2), atol=1e-5)