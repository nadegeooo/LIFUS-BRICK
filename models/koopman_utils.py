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
    u1 = torch.rand(M)
    nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))

    # --- Phase: uniform in [0, 2*pi] ---
    # theta_log = log(theta) so that exp(theta_log) = theta.
    # Clamp away from 0 so log(theta) is finite.
    u2 = torch.rand(M).clamp(min=1e-8)
    theta_log = torch.log(2 * math.pi * u2)

    # Random complex eigenvector matrix P (needs to be invertible; test checks this)
    P = torch.complex(torch.randn(M, M), torch.randn(M, M))

    return nu_log, theta_log, P


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
    return torch.exp(torch.complex(log_magnitude, phase))


# ================================================================================
# SEQUENTIAL SCAN (test utility only — do not use in model)
# ================================================================================

def sequential_scan(
    Lambda: torch.Tensor,
    g0: torch.Tensor,
    n_steps: int
) -> torch.Tensor:
    """
    Propagate latent state g0 forward n_steps using a sequential for-loop.

    Since Lambda is diagonal, the recurrence is elementwise:
        g_t = Lambda^t * g0  (elementwise complex multiply)

    Used only to verify parallel_scan correctness in tests.
    Do NOT use in the model — O(T) sequential steps.

    Args:
        Lambda  (torch.Tensor): Complex eigenvalue vector, shape (M,)
        g0      (torch.Tensor): Initial latent state, shape (M,), complex
        n_steps (int):          Number of timesteps T

    Returns:
        torch.Tensor: Complex latent trajectory of shape (T, M)
    """
    trajectory = []
    g = g0.clone()
    for _ in range(n_steps):
        g = Lambda * g
        trajectory.append(g)
    return torch.stack(trajectory, dim=0)  # (T, M)


# ================================================================================
# PARALLEL SCAN
# ================================================================================

def parallel_scan(
    Lambda: torch.Tensor,
    g0: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """
    Propagate latent state g0 forward n_steps in the eigenspace.
 
    Since Lambda is diagonal, g_t = Lambda^t * g0 (elementwise). The powers
    Lambda^1, ..., Lambda^T are obtained with a single cumulative product
    (a parallel-scan primitive: O(T) work and O(log T) depth on GPU), which
    avoids the previous per-timestep / per-bit Python loop.
 
    Returns row t-1 = Lambda^t * g0, matching sequential_scan.
 
    Note: for this *autonomous* recurrence the powers are a closed form, so a
    scan is not strictly required (Lambda ** t would also work). The cumprod
    formulation is kept because it is the natural building block once the
    control term C @ u_t is folded in (the associative scan over (a, b) pairs).
 
    Args:
        Lambda  (torch.Tensor): Complex eigenvalue vector, shape (M,)
        g0      (torch.Tensor): Initial latent state, shape (M,), complex
        n_steps (int):          Number of timesteps T
 
    Returns:
        torch.Tensor: Complex latent trajectory of shape (T, M),
                      where entry [t] = Lambda^(t+1) * g0
    """
    # (1, M) -> (T, M) view, then cumulative product down the time axis.
    Lambda_rows = Lambda.unsqueeze(0).expand(n_steps, -1)   # (T, M)
    Lambda_powers = torch.cumprod(Lambda_rows, dim=0)       # (T, M): row t-1 = Lambda^t
    return Lambda_powers * g0                               # broadcast (T, M) * (M,)


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
    fc = (x.mT @ x) / (x.shape[-2] - 1)      # works for both (T,N) and (B,T,N)
    return fc