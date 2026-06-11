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

    Eigenvalue magnitudes are log-uniformly sampled in [r_min, r_max] to give
    a spread of memory timescales, biased toward long memory (values near 1).
    Phases are uniformly sampled in [0, 2*pi] to cover all oscillation
    frequencies.

    Args:
        M (int): Latent dimension (N_ROIS x H)

    Returns:
        nu_log    (torch.Tensor): Real vector of shape (M,), magnitude params
        theta_log (torch.Tensor): Real vector of shape (M,), phase params
        P         (torch.Tensor): Complex matrix of shape (M, M), eigenvectors
    """

    # We want samples of r in [r_min, r_max] with log-uniform distribution.
    # To get this, we sample uniformly from [0,1)
    # ]

    # Sample log-uniform magnitudes in [r_min, r_max]
    # log(-log(r)) inverts the stable exponential: exp(-exp(nu)) = r
    u1 = torch.rand(M)
    log_r = u1 * (math.log(r_max) - math.log(r_min)) + math.log(r_min)
    # nu_log = log(-log_r) so that exp(-exp(nu_log)) = r
    nu_log = torch.log(-log_r)

    # Sample uniform phases in [0, 2*pi]
    # theta_log = log(theta) so that exp(theta_log) = theta
    u2 = torch.rand(M).clamp(min=1e-8) #set a minimum so we are not taking log(0)
    theta = 2 * math.pi * u2
    theta_log = torch.log(theta)

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
    n_steps: int
) -> torch.Tensor:
    """
    Propagate latent state g0 forward n_steps using an associative parallel scan.

    Since Lambda is diagonal, g_t = Lambda^t * g0 (elementwise). This means
    we can precompute Lambda^1, Lambda^2, Lambda^4, ... via repeated elementwise
    squaring (O(log T) steps), then assemble the full trajectory in parallel by
    binary decomposition of each timestep index.

    This is O(T log T) elementwise operations vs O(T) for sequential, but
    all T timesteps can be computed in parallel on GPU in O(log T) passes.

    Args:
        Lambda  (torch.Tensor): Complex eigenvalue vector, shape (M,)
        g0      (torch.Tensor): Initial latent state, shape (M,), complex
        n_steps (int):          Number of timesteps T

    Returns:
        torch.Tensor: Complex latent trajectory of shape (T, M),
                      where entry [t] = Lambda^(t+1) * g0
    """
    if n_steps == 1:
        return (Lambda * g0).unsqueeze(0)  # (1, M)

    # Precompute Lambda^1, Lambda^2, Lambda^4, ..., Lambda^(2^ceil(log2(T)))
    # Each is just elementwise squaring — cheap since Lambda is a vector
    max_power = math.ceil(math.log2(n_steps))
    powers = [Lambda]
    current = Lambda
    for _ in range(max_power):
        current = current * current  # Lambda^(2^k) elementwise
        powers.append(current)

    # Build trajectory: g_t = Lambda^t * g0 for t = 1, ..., n_steps
    # Decompose each t in binary and combine precomputed powers elementwise
    trajectory = []
    for t in range(1, n_steps + 1):
        # Compute Lambda^t via binary decomposition
        Lambda_t = torch.ones_like(Lambda)
        bit = 0
        temp = t
        while temp > 0:
            if temp & 1:
                Lambda_t = powers[bit] * Lambda_t  # elementwise
            temp >>= 1
            bit += 1
        trajectory.append(Lambda_t * g0)

    return torch.stack(trajectory, dim=0)  # (T, M)