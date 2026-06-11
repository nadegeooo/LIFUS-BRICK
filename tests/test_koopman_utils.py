import math
import time
import pytest
import torch

from config import M, T as T_DATA, R_MIN, R_MAX

# --- CONSTANTS ---
TOLERANCE_SCAN  = 1e-5
TOLERANCE_EIGEN = 1e-4
T_VALUES        = [1, 2, 3, 4, 7, 8, 9, 16, 17, 50, T_DATA] # Powers of two and neighbors added on purpose
#[1, 10, 50, T_DATA]  # runs quicker


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
# Proves both methods agree with each other, does not prove they are correct.
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


# ================================================================================
# 10. EIGENVALUE MAGNITUDE BIAS TOWARD LONG MEMORY
# ================================================================================
def test_lambda_magnitude_median_above_midpoint():
    """
    Median eigenvalue magnitude must be above the midpoint of [r_min, r_max],
    confirming the initialization is biased toward long memory (values near r_max).
    Ring-uniform sampling (uniform in |Lambda|^2) achieves this.
    Log-uniform sampling (uniform in log r) does NOT — it concentrates mass
    near r_min (short memory).
    """
    from models.koopman_utils import init_koopman_params, compute_lambda
    from config import R_MIN, R_MAX

    torch.manual_seed(42)

    # Use many samples for statistical reliability
    all_magnitudes = []
    for _ in range(100):
        nu_log, theta_log, _ = init_koopman_params(M)
        Lambda = compute_lambda(nu_log, theta_log)
        all_magnitudes.append(Lambda.abs())

    magnitudes = torch.cat(all_magnitudes)
    median = magnitudes.median().item()
    midpoint = (R_MIN + R_MAX) / 2

    assert median > midpoint, \
        f"Median magnitude {median:.4f} is below midpoint {midpoint:.4f} — " \
        f"initialization is biased toward short memory, not long memory"
    


# ================================================================================
# GROUND TRUTH TEST: scans vs an INDEPENDENT oracle (not each other)
# ================================================================================
# For parallel vs sequence
@pytest.mark.parametrize("n_steps", T_VALUES)
def test_scans_match_closed_form(n_steps):
    """
    g_t = Lambda^t * g0 has the closed form Lambda**t * g0, computed here via
    exp(t*log Lambda) -- a different code path from both iterated-multiply
    (sequential) and cumprod (parallel). This pins the 'entry[t] = Lambda^(t+1)*g0'
    convention against a true ground truth, not against the other scan.
    """
    from models.koopman_utils import (
        init_koopman_params, compute_lambda, parallel_scan, sequential_scan,
    )
 
    torch.manual_seed(0)
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
 
    t = torch.arange(1, n_steps + 1).unsqueeze(1)             # (T, 1)
    expected = (Lambda.unsqueeze(0) ** t) * g0                # (T, M) closed form
 
    out_par = parallel_scan(Lambda, g0, n_steps)
    out_seq = sequential_scan(Lambda, g0, n_steps)
 
    assert torch.allclose(out_par, expected, atol=TOLERANCE_SCAN), \
        f"parallel vs closed-form max diff {(out_par - expected).abs().max().item()}"
    assert torch.allclose(out_seq, expected, atol=TOLERANCE_SCAN), \
        f"sequential vs closed-form max diff {(out_seq - expected).abs().max().item()}"
 
 
def test_scan_hand_worked_real():
    """Tiny human-verifiable case (non-stable Lambda is fine here -- we build it
    directly and never call compute_lambda)."""
    from models.koopman_utils import parallel_scan, sequential_scan
 
    Lambda = torch.tensor([2.0, 3.0], dtype=torch.complex64)
    g0 = torch.tensor([1.0, 1.0], dtype=torch.complex64)
    expected = torch.tensor([[2, 3], [4, 9], [8, 27]], dtype=torch.complex64)
 
    assert torch.allclose(parallel_scan(Lambda, g0, 3), expected)
    assert torch.allclose(sequential_scan(Lambda, g0, 3), expected)
 
 
def test_scan_hand_worked_complex():
    """Tiny complex case: Lambda=[i, 0.5], g0=[1, 2] ->
    [i,1], [-1,0.5], [-i,0.25]. Exercises complex multiply + the phase sign."""
    from models.koopman_utils import parallel_scan, sequential_scan
 
    Lambda = torch.tensor([1j, 0.5 + 0j], dtype=torch.complex64)
    g0 = torch.tensor([1.0 + 0j, 2.0 + 0j], dtype=torch.complex64)
    expected = torch.tensor(
        [[1j, 1.0], [-1.0, 0.5], [-1j, 0.25]], dtype=torch.complex64
    )
 
    assert torch.allclose(parallel_scan(Lambda, g0, 3), expected, atol=1e-6)
    assert torch.allclose(sequential_scan(Lambda, g0, 3), expected, atol=1e-6)
 
 
# ================================================================================
# EXACT FORMULA: catches a sign flip on the phase term
# ================================================================================
def test_compute_lambda_exact_formula():
    """
    compute_lambda must equal exp(-exp(nu) + i*exp(theta)) with the correct
    sign on the imaginary part. A conjugated implementation (-i*exp(theta))
    has identical magnitude and passes every stability test, so only an exact
    check with the right sign catches it.
    """
    nu_log = torch.tensor([-0.5, 0.0, 0.3])
    theta_log = torch.tensor([0.1, -0.2, 0.4])
 
    from models.koopman_utils import compute_lambda
    got = compute_lambda(nu_log, theta_log)
 
    expected = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
    assert torch.allclose(got, expected, atol=1e-6), "compute_lambda formula mismatch"
 
    # Explicit anchor: phase = pi/2 must give a purely imaginary +i*|Lambda|.
    nu_log = torch.tensor([math.log(-math.log(0.9))])
    theta_log = torch.tensor([math.log(math.pi / 2)])
    lam = compute_lambda(nu_log, theta_log)[0]
    assert lam.imag.item() > 0, "phase pi/2 should give POSITIVE imaginary part (sign flip?)"
    assert abs(lam.real.item()) < 1e-5, "phase pi/2 should give ~zero real part"
 
 
# ================================================================================
# INIT: magnitudes land inside the requested ring [r_min, r_max]
# ================================================================================
def test_init_magnitudes_within_ring():
    """
    At initialization, |Lambda| must lie in [r_min, r_max] (ring sampling),
    a tighter contract than the (0,1) stability bound. NOTE: this holds only
    at init -- training may move magnitudes anywhere inside (0,1).
    """
    from models.koopman_utils import init_koopman_params, compute_lambda
 
    torch.manual_seed(0)
    mags = torch.cat([
        compute_lambda(*init_koopman_params(M)[:2]).abs() for _ in range(50)
    ])
    tol = 1e-4
    assert (mags >= R_MIN - tol).all(), f"min |Lambda| {mags.min().item():.4f} < r_min {R_MIN}"
    assert (mags <= R_MAX + tol).all(), f"max |Lambda| {mags.max().item():.4f} > r_max {R_MAX}"
 
 
# ================================================================================
# GRADIENTS: parameters must remain trainable through the eigenspace path
# ================================================================================
def test_gradients_flow_through_scan():
    """
    nu_log and theta_log are trained parameters. A value-only suite never checks
    that gradients flow; an accidental .detach()/in-place op/broken complex
    autograd would freeze them silently. (P is not used by these functions, so it
    is out of scope here.)
    """
    from models.koopman_utils import compute_lambda, parallel_scan
 
    torch.manual_seed(0)
    nu_log = torch.randn(M, requires_grad=True)
    theta_log = torch.randn(M, requires_grad=True)
    g0 = torch.randn(M, dtype=torch.complex64)
 
    Lambda = compute_lambda(nu_log, theta_log)
    out = parallel_scan(Lambda, g0, n_steps=10)
    out.abs().sum().backward()
 
    for name, p in [("nu_log", nu_log), ("theta_log", theta_log)]:
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is all zero"
 
 
# ================================================================================
# RUNTIME
# ================================================================================
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="parallel scan only beats the loop where it can actually "
                           "parallelize (GPU); on CPU both are O(T) work and the "
                           "loop can win.")
def test_parallel_faster_than_sequential_gpu():
    """Where it is meaningful (GPU), parallel_scan should be clearly faster."""
    from models.koopman_utils import (
        init_koopman_params, compute_lambda, parallel_scan, sequential_scan,
    )
 
    dev = "cuda"
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log).to(dev)
    g0 = torch.randn(M, dtype=torch.complex64, device=dev)
    T = max(4096, T_DATA)
 
    def best(fn, reps=5):
        fn(); torch.cuda.synchronize()                     # warm up
        times = []
        for _ in range(reps):
            torch.cuda.synchronize(); s = time.perf_counter()
            fn(); torch.cuda.synchronize()
            times.append(time.perf_counter() - s)
        return min(times)
 
    t_par = best(lambda: parallel_scan(Lambda, g0, T))
    t_seq = best(lambda: sequential_scan(Lambda, g0, T))
    assert t_par < 0.5 * t_seq, f"parallel {t_par*1e3:.1f}ms not < half of sequential {t_seq*1e3:.1f}ms"
 
 
def test_parallel_scan_runtime_ceiling():
    """
    CPU-safe regression guard: a large T must finish under a generous bound.
    Catches a catastrophic blow-up (e.g. accidental O(T^2)) WITHOUT racing the
    two implementations, which is unreliable on CPU.
    """
    from models.koopman_utils import init_koopman_params, compute_lambda, parallel_scan
 
    nu_log, theta_log, _ = init_koopman_params(M)
    Lambda = compute_lambda(nu_log, theta_log)
    g0 = torch.randn(M, dtype=torch.complex64)
    T = max(10000, T_DATA)
 
    parallel_scan(Lambda, g0, T)                            # warm up
    s = time.perf_counter()
    parallel_scan(Lambda, g0, T)
    elapsed = time.perf_counter() - s
    assert elapsed < 2.0, f"parallel_scan took {elapsed:.2f}s for T={T} (regression?)"