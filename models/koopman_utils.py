"""
================================================================================
Koopman Operator Utilities
================================================================================

Description:
    Core mathematical utilities for the Koopman linearisation component of BRICK.

    Rather than training K directly, BRICK trains its eigendecomposition:
        K = P @ diag(Lambda) @ P_inv

    where P is a complex eigenvector matrix and Lambda is a complex diagonal
    vector of eigenvalues. The forward pass never forms K explicitly — instead
    it operates in the eigenspace using the parallel scan algorithm.

    Eigenvalues are parameterized via the stable exponential form from the LRU
    paper (Orvieto et al. 2023):
        Lambda_j = exp(-exp(nu_log_j) + i * exp(theta_log_j))

    This guarantees |Lambda_j| < 1 for all values of nu_log and theta_log,
    including after gradient updates — stability is enforced by the
    parameterization, not by post-hoc normalization.

Functions:
    init_koopman_params(M)              -- Initialize nu_log, theta_log, P
    compute_lambda(nu_log, theta_log)   -- Compute complex eigenvalues from params
    sequential_scan(Lambda, g0, n_steps) -- Naive for-loop scan (test utility only)
    parallel_scan(Lambda, g0, n_steps)  -- Associative parallel scan (used in model)

References:
    - Zhou et al. 2025 (BRICK paper), Section III-A
    - Orvieto et al. 2023 (LRU paper), Section 3
"""

import math
import torch

from config import R_MIN, R_MAX


# ================================================================================
# INITIALISATION
# ================================================================================

def init_koopman_params(M: int, r_min: float = R_MIN, r_max: float = R_MAX):
    """
    Initialize Koopman operator parameters using LRU-style eigenvalue
    initialization.

    Returns nu_log and theta_log (real vectors) which parameterize the
    eigenvalues via the stable exponential form, and P (complex matrix)
    which defines the eigenvector directions.

    Eigenvalue magnitudes are ring-uniformly sampled (uniform in |Lambda|^2) in
    [r_min, r_max], biased toward long memory (values near r_max). This matches
    the initialization method from Orvieto et al. 2023 (LRU paper).

    Args:
        M (int): Latent dimension (N_ROIS x H)

    Returns:
        nu_log    (torch.Tensor): Real vector of shape (M,), magnitude params
        theta_log (torch.Tensor): Real vector of shape (M,), phase params
        P         (torch.Tensor): Complex matrix of shape (M, M), eigenvectors
    """

    # Ring-uniform: sample |Lambda|^2 uniformly in [r_min^2, r_max^2],
    # then take the square root to get r. This biases magnitudes toward r_max
    # (long memory), matching the LRU paper initialization.

    # Sample log-uniform magnitudes in [r_min, r_max]
    # log(-log(r)) inverts the stable exponential: exp(-exp(nu)) = r
    u1 = torch.rand(M)  #[0,1), uniform for ring-uniform sampling
    nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))

    # --- Phase: uniform in [0, 2*pi] ---
    # theta_log = log(theta) so that exp(theta_log) = theta.
    # Clamp away from 0 so log(theta) is finite.
    u2 = torch.rand(M).clamp(min=1e-8)
    theta_log = torch.log(2 * math.pi * u2)

    # Random complex eigenvector matrix P (needs to be invertible; test checks this)
    P_inv = torch.complex(torch.randn(M, M), torch.randn(M, M))         # Creates 2 MxM matrices and merges them so the first is real the second is imaginary.

    return nu_log, theta_log, P_inv


# ================================================================================
# EIGENVALUE COMPUTATION
# ================================================================================

def compute_lambda(nu_log: torch.Tensor, theta_log: torch.Tensor) -> torch.Tensor:
    """
    Compute complex eigenvalues from the stable exponential parameterization.

    Formula (from LRU paper):
        Lambda_j = exp(-exp(nu_log_j) + i * exp(theta_log_j))

    The magnitude is exp(-exp(nu_log_j)), which is always in (0, 1) for any
    real value of nu_log_j — this guarantees stability throughout training.

    Args:
        nu_log    (torch.Tensor): Real vector of shape (M,)
        theta_log (torch.Tensor): Real vector of shape (M,)

    Returns:
        torch.Tensor: Complex eigenvalue vector of shape (M,)
    """
    log_magnitude = -torch.exp(nu_log)      # always negative -> magnitude in (0,1)
    phase         =  torch.exp(theta_log)   # always positive
    return torch.exp(torch.complex(log_magnitude, phase)) #lambda


# ================================================================================
# SEQUENTIAL SCAN (test utility only — do not use in model)
# ================================================================================

def sequential_scan(
    Lambda: torch.Tensor,
    u_bar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute linear recurrence sequentially (test utility only — do not use in model):
        ḡ_t = Λ ⊙ ḡ_{t-1} + ū_t,  ḡ_0 = 0
 
    Args:
        Lambda (torch.Tensor): Complex eigenvalues, shape (M,)
        u_bar  (torch.Tensor): Transformed input sequence, shape (T, M), complex
                               u_bar[0]  = P_inv @ g_0      (initial condition)
                               u_bar[1:] = P_inv @ C @ u_t  (control inputs)
 
    Returns:
        torch.Tensor: Latent trajectory in eigenspace, shape (T, M), complex
    """
    trajectory = []
    g_bar = torch.zeros_like(u_bar[0])
    for t in range(u_bar.shape[0]):
        g_bar = Lambda * g_bar + u_bar[t]
        trajectory.append(g_bar)
    return torch.stack(trajectory, dim=0)  # (T, M)



# ================================================================================
# PARALLEL SCAN
# ================================================================================

def parallel_scan(
    Lambda: torch.Tensor,
    u_bar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute linear recurrence using associative parallel scan:
        ḡ_t = Λ ⊙ ḡ_{t-1} + ū_t,  ḡ_0 = 0
 
    Uses the binary operator from Orvieto et al. 2023 (LRU paper):
        (a_i, b_i) ⊕ (a_j, b_j) = (a_j * a_i, a_j * b_i + b_j)
 
    Implemented via recursive doubling: O(T log T) work, O(log T) depth.
 
    Args:
        Lambda (torch.Tensor): Complex eigenvalues, shape (M,)
        u_bar  (torch.Tensor): Transformed input sequence, shape (T, M), complex
                               u_bar[0]  = P_inv @ g_0      (initial condition)
                               u_bar[1:] = P_inv @ C @ u_t  (control inputs)
 
    Returns:
        torch.Tensor: Latent trajectory in eigenspace, shape (T, M), complex
    """
    T, M = u_bar.shape
 
    # Expand Lambda to (T, M) — same eigenvalues broadcast at every timestep
    a = Lambda.unsqueeze(0).expand(T, -1).clone()  # (T, M)
    b = u_bar.clone()                               # (T, M)
 
    # Pad to next power of 2 for clean recursive doubling
    next_pow2 = 1
    while next_pow2 < T:
        next_pow2 *= 2
 
    pad = next_pow2 - T
    if pad > 0:
        a = torch.cat([a, torch.ones(pad, M, dtype=a.dtype, device=a.device)],  dim=0)
        b = torch.cat([b, torch.zeros(pad, M, dtype=b.dtype, device=b.device)], dim=0)
 
    # Inclusive prefix scan via recursive doubling
    # After each step, element i contains combined result of [i - step .. i]
    step = 1
    while step < next_pow2:
        i = torch.arange(step, next_pow2, device=u_bar.device)
        j = i - step
        a_new = a[i] * a[j]
        b_new = a[i] * b[j] + b[i]
        a = a.clone()
        b = b.clone()
        a[i] = a_new
        b[i] = b_new
        step *= 2
 
    return b[:T]  # (T, M) — remove padding


# ================================================================================
# PEARSON CORRELATION
# ================================================================================

def compute_pearson(x: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation matrix from timeseries.
    
    Args:
        x (torch.Tensor): shape (T, N) — timepoints x ROIs
    
    Returns:
        torch.Tensor: shape (N, N) — FC matrix
    """
    # Centre each ROI (subtract mean)
    x = x - x.mean(dim=-2, keepdim=True)
    # Normalise each ROI by its std
    std = x.std(dim=-2, keepdim=True).clamp(min=1e-8)
    x = x / std
    # Correlation = (X^T X) / (T - 1)
    fc = (x.mT @ x) / (x.shape[-2] - 1)      # works for both (T,N) and (B,T,N): makes it batch aware
    return fc