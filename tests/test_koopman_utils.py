"""
================================================================================
Tests for models/koopman_utils.py
================================================================================

Covers:
    - init_koopman_params: shapes, types
    - compute_lambda: formula correctness, stability, ring-uniform bias
    - parallel_scan / sequential_scan: shape, correctness vs closed form,
      hand-worked cases, edge cases, gradient flow, GPU runtime
    - compute_pearson: shape, symmetry, value range, edge cases

Note: Pearson tests are here because compute_pearson lives in koopman_utils.
      If it is moved to models/utils.py these tests should follow.
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

# --- CONSTANTS ---
TOLERANCE_SCAN  = 1e-4
TOLERANCE_EIGEN = 1e-4
# Powers of two and neighbours included deliberately to stress boundary behaviour
T_VALUES = [1, 2, 3, 4, 7, 8, 9, 16, 17, 50, T_DATA]


# ================================================================================
# 1. init_koopman_params
# ================================================================================

def test_init_params_shapes():
    """nu_log and theta_log must be (M,); P must be (M, M)."""
    nu_log, theta_log, P = init_koopman_params(M)

    assert nu_log.shape    == (M,),   f"nu_log: expected ({M},), got {nu_log.shape}"
    assert theta_log.shape == (M,),   f"theta_log: expected ({M},), got {theta_log.shape}"
    assert P.shape         == (M, M), f"P: expected ({M}, {M}), got {P.shape}"


def test_init_params_types():
    """nu_log and theta_log must be real; P must be complex."""
    nu_log, theta_log, P = init_koopman_params(M)

    assert not nu_log.is_complex(),    "nu_log should be real"
    assert not theta_log.is_complex(), "theta_log should be real"
    assert P.is_complex(),             "P should be complex"


def test_P_is_invertible():
    """P must be invertible — full rank."""
    _, _, P = init_koopman_params(M)
    rank = torch.linalg.matrix_rank(P).item()
    assert rank == M, f"P is not full rank: rank={rank}, expected {M}"


def test_init_magnitudes_within_ring():
    """
    At initialisation, |Lambda| must lie in [r_min, r_max].
    This is a tighter contract than the (0, 1) stability bound and holds only
    at init — training is free to move magnitudes anywhere inside (0, 1).
    """
    torch.manual_seed(0)
    mags = torch.cat([
        compute_lambda(*init_koopman_params(M)[:2]).abs() for _ in range(50)
    ])

    assert (mags >= R_MIN - TOLERANCE_EIGEN).all(), \
        f"min |Lambda| {mags.min().item():.4f} < r_min {R_MIN}"
    assert (mags <= R_MAX + TOLERANCE_EIGEN).all(), \
        f"max |Lambda| {mags.max().item():.4f} > r_max {R_MAX}"


def test_lambda_magnitude_median_above_midpoint():
    """
    Median |Lambda| must be above the midpoint of [r_min, r_max], confirming
    ring-uniform sampling (uniform in |Lambda|^2) biases toward long memory.
    Log-uniform sampling would give the opposite bias.
    """
    torch.manual_seed(42)
    all_mags = []
    for _ in range(100):
        nu_log, theta_log, _ = init_koopman_params(M)
        all_mags.append(compute_lambda(nu_log, theta_log).abs())

    magnitudes = torch.cat(all_mags)
    median     = magnitudes.median().item()
    midpoint   = (R_MIN + R_MAX) / 2

    assert median > midpoint, (
        f"Median |Lambda| {median:.4f} is below midpoint {midpoint:.4f} — "
        "initialisation is biased toward short memory, not long memory"
    )


# ================================================================================
# 2. compute_lambda
# ================================================================================

def test_lambda_shape():
    """Lambda must be a complex vector of shape (M,)."""
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)

    assert Lambda.shape == (M,), f"Expected ({M},), got {Lambda.shape}"
    assert Lambda.is_complex(),  "Lambda must be complex"


def test_lambda_magnitudes_inside_unit_circle():
    """All |Lambda| must be strictly in (0, 1)."""
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    mags   = Lambda.abs()

    assert (mags > 0).all(), \
        "Eigenvalue magnitudes must be > 0"
    assert (mags < 1).all(), \
        f"Eigenvalue magnitudes must be < 1, got max {mags.max().item():.6f}"


def test_lambda_stable_after_perturbation():
    """
    |Lambda| must remain in (0, 1) after a gradient-like perturbation of
    nu_log and theta_log, verifying that stability is enforced by the
    parameterisation rather than post-hoc normalisation.
    """
    nu_log, theta_log, _ = init_koopman_params(M)
    nu_log    = nu_log    + 0.1 * torch.randn(M)
    theta_log = theta_log + 0.1 * torch.randn(M)

    mags = compute_lambda(nu_log, theta_log).abs()

    assert (mags > 0).all(), "Magnitudes must be > 0 after perturbation"
    assert (mags < 1).all(), \
        f"Magnitudes must be < 1 after perturbation, got max {mags.max().item():.6f}"


def test_compute_lambda_exact_formula():
    """
    compute_lambda must implement exp(-exp(nu) + i*exp(theta)) exactly.
    A conjugated implementation (-i*exp(theta)) has identical magnitude and
    passes all stability tests, so only an exact formula check catches it.
    Also anchors the sign: phase = pi/2 must give +i*|Lambda|, not -i*|Lambda|.
    """
    nu_log    = torch.tensor([-0.5, 0.0,  0.3])
    theta_log = torch.tensor([ 0.1, -0.2, 0.4])

    got      = compute_lambda(nu_log, theta_log)
    expected = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
    assert torch.allclose(got, expected, atol=1e-6), "compute_lambda formula mismatch"

    nu_log_anchor    = torch.tensor([math.log(-math.log(0.9))])
    theta_log_anchor = torch.tensor([math.log(math.pi / 2)])
    lam = compute_lambda(nu_log_anchor, theta_log_anchor)[0]
    assert lam.imag.item() > 0,        "phase pi/2 should give POSITIVE imaginary part"
    assert abs(lam.real.item()) < 1e-5, "phase pi/2 should give ~zero real part"


# ================================================================================
# 3. parallel_scan and sequential_scan
# ================================================================================

@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_output_shape(n_steps):
    """parallel_scan output must be (T, M)."""
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0  = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, g0, n_steps)

    assert out.shape == (n_steps, M), f"Expected ({n_steps}, {M}), got {out.shape}"


@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_matches_sequential(n_steps):
    """
    parallel_scan must agree with the sequential for-loop within TOLERANCE_SCAN.
    Note: this verifies consistency between the two implementations, not
    correctness against an independent oracle — see test_scans_match_closed_form.
    """
    torch.manual_seed(42)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0     = torch.randn(M, dtype=torch.complex64)

    out_par = parallel_scan(Lambda, g0, n_steps)
    out_seq = sequential_scan(Lambda, g0, n_steps)

    assert torch.allclose(out_par, out_seq, atol=TOLERANCE_SCAN), \
        f"Max difference: {(out_par - out_seq).abs().max().item()}"


@pytest.mark.parametrize("n_steps", T_VALUES)
def test_scans_match_closed_form(n_steps):
    """
    Both scans must match the closed form Lambda**t * g0, computed via
    elementwise power — a different code path from both cumprod (parallel)
    and iterated multiply (sequential). This is the independent oracle.
    """
    torch.manual_seed(0)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0     = torch.randn(M, dtype=torch.complex64)

    t        = torch.arange(1, n_steps + 1).unsqueeze(1)  # (T, 1)
    expected = (Lambda.unsqueeze(0) ** t) * g0             # (T, M)

    assert torch.allclose(parallel_scan(Lambda, g0, n_steps), expected, atol=TOLERANCE_SCAN), \
        f"parallel vs closed-form max diff {(parallel_scan(Lambda, g0, n_steps) - expected).abs().max().item()}"
    assert torch.allclose(sequential_scan(Lambda, g0, n_steps), expected, atol=TOLERANCE_SCAN), \
        f"sequential vs closed-form max diff {(sequential_scan(Lambda, g0, n_steps) - expected).abs().max().item()}"


def test_parallel_scan_edge_case_T1():
    """parallel_scan must handle T=1 and return shape (1, M)."""
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0  = torch.randn(M, dtype=torch.complex64)
    out = parallel_scan(Lambda, g0, n_steps=1)

    assert out.shape == (1, M), f"Expected (1, {M}), got {out.shape}"


def test_parallel_scan_deterministic():
    """Same inputs must always produce the same output."""
    torch.manual_seed(42)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0     = torch.randn(M, dtype=torch.complex64)

    assert torch.allclose(
        parallel_scan(Lambda, g0, n_steps=20),
        parallel_scan(Lambda, g0, n_steps=20),
    ), "parallel_scan is not deterministic"


@pytest.mark.parametrize("n_steps", T_VALUES)
def test_parallel_scan_no_nan_inf(n_steps):
    """parallel_scan must not produce NaN or Inf values."""
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0     = torch.randn(M, dtype=torch.complex64)
    out    = parallel_scan(Lambda, g0, n_steps)

    assert torch.isfinite(out.abs()).all(), \
        f"NaN or Inf in parallel_scan output at T={n_steps}"


def test_parallel_scan_stable():
    """Implicityly tests that Lambda**t does not explode in magnitude, which would indicate a stability violation.
    Should hold anyways because of limits on lambda"""
    torch.manual_seed(42)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0     = torch.randn(M, dtype=torch.complex64)
    out    = parallel_scan(Lambda, g0, n_steps=T_DATA)

    initial_norm = g0.abs().norm().item()
    final_norm   = out[-1].abs().norm().item()

    assert final_norm <= initial_norm, (
        f"Latent state exploded: initial norm={initial_norm:.3f}, "
        f"final norm={final_norm:.3f}"
    )


def test_scan_hand_worked_real():
    """
    Human-verifiable case with real Lambda=[2, 3], g0=[1, 1]:
        t=1: [2, 3]   t=2: [4, 9]   t=3: [8, 27]
    Lambda magnitudes > 1 are intentional here — we bypass compute_lambda
    to test the scan arithmetic directly.
    """
    Lambda   = torch.tensor([2.0, 3.0], dtype=torch.complex64)
    g0       = torch.tensor([1.0, 1.0], dtype=torch.complex64)
    expected = torch.tensor([[2, 3], [4, 9], [8, 27]], dtype=torch.complex64)

    assert torch.allclose(parallel_scan(Lambda, g0, 3), expected)
    assert torch.allclose(sequential_scan(Lambda, g0, 3), expected)


def test_scan_hand_worked_complex():
    """
    Human-verifiable case with complex Lambda=[i, 0.5], g0=[1, 2]:
        t=1: [i, 1]   t=2: [-1, 0.5]   t=3: [-i, 0.25]
    Exercises complex multiply and verifies the phase sign convention.
    """
    Lambda   = torch.tensor([1j, 0.5 + 0j], dtype=torch.complex64)
    g0       = torch.tensor([1.0 + 0j, 2.0 + 0j], dtype=torch.complex64)
    expected = torch.tensor(
        [[1j, 1.0], [-1.0, 0.5], [-1j, 0.25]], dtype=torch.complex64
    )

    assert torch.allclose(parallel_scan(Lambda, g0, 3), expected, atol=1e-6)
    assert torch.allclose(sequential_scan(Lambda, g0, 3), expected, atol=1e-6)


def test_gradients_flow_through_scan():
    """
    nu_log and theta_log must receive non-zero finite gradients through the
    scan. An accidental detach() or in-place op would freeze them silently.
    P is not used by the scan functions and is out of scope here.
    """
    torch.manual_seed(0)
    nu_log    = torch.randn(M, requires_grad=True)
    theta_log = torch.randn(M, requires_grad=True)
    g0        = torch.randn(M, dtype=torch.complex64)

    Lambda = compute_lambda(nu_log, theta_log)
    out    = parallel_scan(Lambda, g0, n_steps=10)
    out.abs().sum().backward()

    for name, p in [("nu_log", nu_log), ("theta_log", theta_log)]:
        assert p.grad is not None,              f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(),    f"{name} has non-finite gradient"
        assert p.grad.abs().sum() > 0,          f"{name} gradient is all zero"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason=(
        "parallel_scan only outperforms the loop where it can actually parallelise "
        "(GPU). On CPU both are O(T) work and the loop may win."
    ),
)
def test_parallel_faster_than_sequential_gpu():
    """On GPU, parallel_scan must be at least 2x faster than sequential_scan."""
    dev    = "cuda"
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log).to(dev)
    g0     = torch.randn(M, dtype=torch.complex64, device=dev)
    T      = max(4096, T_DATA)

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

    t_par = best(lambda: parallel_scan(Lambda, g0, T))
    t_seq = best(lambda: sequential_scan(Lambda, g0, T))
    assert t_par < 0.5 * t_seq, (
        f"parallel {t_par*1e3:.1f}ms not < half of sequential {t_seq*1e3:.1f}ms"
    )


# ================================================================================
# 4. compute_pearson
# ================================================================================

def test_pearson_output_shape():
    """FC matrix must be (N, N)."""
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert fc.shape == (N_ROIS, N_ROIS), \
        f"Expected ({N_ROIS}, {N_ROIS}), got {fc.shape}"


def test_pearson_diagonal_is_one():
    """Self-correlation must be 1.0."""
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert torch.allclose(torch.diag(fc), torch.ones(N_ROIS), atol=1e-5), \
        f"Diagonal is not 1.0: {torch.diag(fc)}"


def test_pearson_symmetric():
    """FC matrix must be symmetric."""
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert torch.allclose(fc, fc.T, atol=1e-5), "FC matrix is not symmetric"


def test_pearson_values_in_range():
    """All FC values must be in [-1, 1]."""
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert (fc >= -1.0 - 1e-5).all() and (fc <= 1.0 + 1e-5).all(), \
        f"FC values outside [-1, 1]: min={fc.min().item():.4f}, max={fc.max().item():.4f}"


def test_pearson_no_nan_inf():
    """FC matrix must not contain NaN or Inf."""
    fc = compute_pearson(torch.randn(T_DATA, N_ROIS))
    assert torch.isfinite(fc).all(), "FC matrix contains NaN or Inf"


def test_pearson_permutation_equivariant():
    """Permuting input columns must permute FC rows and columns the same way."""
    torch.manual_seed(42)
    x    = torch.randn(T_DATA, N_ROIS)
    perm = torch.randperm(N_ROIS)

    assert torch.allclose(
        compute_pearson(x)[perm][:, perm],
        compute_pearson(x[:, perm]),
        atol=1e-5,
    ), "FC matrix is not permutation equivariant"


def test_pearson_constant_roi():
    """A constant ROI (zero std) must not produce NaN — clamp guards div/0."""
    x       = torch.randn(T_DATA, N_ROIS)
    x[:, 0] = 0.0
    assert torch.isfinite(compute_pearson(x)).all(), \
        "FC matrix contains NaN or Inf when a ROI is constant"


def test_pearson_deterministic():
    """Same input must always produce the same output."""
    torch.manual_seed(42)
    x = torch.randn(T_DATA, N_ROIS)
    assert torch.allclose(compute_pearson(x), compute_pearson(x)), \
        "compute_pearson is not deterministic"


def test_pearson_perfect_correlation():
    """Two identical ROIs must have correlation exactly 1.0."""
    x       = torch.randn(T_DATA, N_ROIS)
    x[:, 1] = x[:, 0]
    fc      = compute_pearson(x)
    assert abs(fc[0, 1].item() - 1.0) < 1e-5, \
        f"Identical ROIs: correlation {fc[0, 1].item():.6f}, expected 1.0"


def test_pearson_perfect_anticorrelation():
    """Two opposite ROIs must have correlation exactly -1.0."""
    x       = torch.randn(T_DATA, N_ROIS)
    x[:, 1] = -x[:, 0]
    fc      = compute_pearson(x)
    assert abs(fc[0, 1].item() + 1.0) < 1e-5, \
        f"Anticorrelated ROIs: correlation {fc[0, 1].item():.6f}, expected -1.0"


def test_pearson_matches_numpy_oracle():
    """Verify against numpy.corrcoef as an independent ground truth."""
    torch.manual_seed(42)
    x        = torch.randn(T_DATA, N_ROIS)
    fc_torch = compute_pearson(x).numpy()
    fc_numpy = np.corrcoef(x.numpy(), rowvar=False)

    assert np.allclose(fc_torch, fc_numpy, atol=1e-5), \
        f"Max diff from numpy: {np.abs(fc_torch - fc_numpy).max():.2e}"


def test_pearson_hand_worked():
    """
    Hand-worked 2x2 case: x = [[1,2],[3,4],[5,6]].
    Columns are perfectly correlated so all FC values must be 1.0.
    """
    x  = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    fc = compute_pearson(x)
    assert torch.allclose(fc, torch.ones(2, 2), atol=1e-5), \
        f"Expected all 1s for perfectly correlated columns, got {fc}"