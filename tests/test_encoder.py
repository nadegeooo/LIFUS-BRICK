# tests/test_encoder.py

import torch
from config import M, N_ROIS, H, T as T_DATA, NHEAD
from models.encoder import Encoder


# ================================================================================
# 0. CONFIG SANITY
# ================================================================================
def test_H_less_than_N():
    """H must be less than N_ROIS — architectural constraint."""
    assert H < N_ROIS, f"H={H} must be less than N_ROIS={N_ROIS}"


# ================================================================================
# 1. FC MATRIX SHAPE AND PROPERTIES
# ================================================================================
def test_fc_matrix_shape():
    """FC matrix must be (N, N)."""
    from models.koopman_utils import compute_pearson

    x = torch.randn(T_DATA, N_ROIS)
    fc = compute_pearson(x)

    assert fc.shape == (N_ROIS, N_ROIS), \
        f"Expected ({N_ROIS}, {N_ROIS}), got {fc.shape}"


def test_fc_matrix_symmetric():
    """FC matrix must be symmetric."""
    from models.koopman_utils import compute_pearson

    x = torch.randn(T_DATA, N_ROIS)
    fc = compute_pearson(x)

    assert torch.allclose(fc, fc.T, atol=1e-5), "FC matrix is not symmetric"


def test_fc_matrix_diagonal_is_one():
    """FC diagonal must be 1.0."""
    from models.koopman_utils import compute_pearson

    x = torch.randn(T_DATA, N_ROIS)
    fc = compute_pearson(x)

    assert torch.allclose(torch.diag(fc), torch.ones(N_ROIS), atol=1e-5), \
        f"FC diagonal is not 1.0: {torch.diag(fc)}"


def test_fc_matrix_no_nan_inf():
    """FC matrix must not contain NaN or Inf."""
    from models.koopman_utils import compute_pearson

    x = torch.randn(T_DATA, N_ROIS)
    fc = compute_pearson(x)

    assert torch.isfinite(fc).all(), "FC matrix contains NaN or Inf"


# ================================================================================
# 2. ROW-WISE MLP OUTPUT (Z_0)
# ================================================================================
def test_z0_shape():
    """Row-wise MLP output Z_0 must be (N, H)."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    z0 = enc.encode_spatial(x)

    assert z0.shape == (N_ROIS, H), \
        f"Expected ({N_ROIS}, {H}), got {z0.shape}"


def test_z0_no_nan_inf():
    """Z_0 must not contain NaN or Inf."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    z0 = enc.encode_spatial(x)

    assert torch.isfinite(z0).all(), "Z_0 contains NaN or Inf"


def test_z0_is_real():
    """Z_0 must be real-valued."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    z0 = enc.encode_spatial(x)

    assert not z0.is_complex(), "Z_0 must be real-valued"


# ================================================================================
# 3. MU AND LOGVAR SHAPES
# ================================================================================
def test_mu_logvar_shapes():
    """mu and logvar must both be (M,)."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    mu, logvar = enc.encode_distribution(x)

    assert mu.shape == (M,), \
        f"mu shape {mu.shape}, expected ({M},)"
    assert logvar.shape == (M,), \
        f"logvar shape {logvar.shape}, expected ({M},)"


def test_mu_logvar_same_shape():
    """mu and logvar must have the same shape."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    mu, logvar = enc.encode_distribution(x)

    assert mu.shape == logvar.shape, \
        f"mu and logvar shapes differ: {mu.shape} vs {logvar.shape}"


def test_mu_logvar_no_nan_inf():
    """mu and logvar must not contain NaN or Inf."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    mu, logvar = enc.encode_distribution(x)

    assert torch.isfinite(mu).all(),     "mu contains NaN or Inf"
    assert torch.isfinite(logvar).all(), "logvar contains NaN or Inf"


def test_mu_logvar_are_real():
    """mu and logvar must be real-valued."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    mu, logvar = enc.encode_distribution(x)

    assert not mu.is_complex(),     "mu must be real-valued"
    assert not logvar.is_complex(), "logvar must be real-valued"


def test_mu_logvar_different_params():
    """mu and logvar must come from separate transformer heads."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    mu, logvar = enc.encode_distribution(x)

    assert not torch.allclose(mu, logvar), \
        "mu and logvar are identical — they may share weights"


# ================================================================================
# 4. G_0 SAMPLE SHAPE AND PROPERTIES
# ================================================================================
def test_g0_shape():
    """g_0 sample must be (M,)."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    g0, _, _ = enc(x)

    assert g0.shape == (M,), f"Expected ({M},), got {g0.shape}"


def test_g0_is_real():
    """g_0 must be real-valued."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    g0, _, _ = enc(x)

    assert not g0.is_complex(), "g_0 must be real-valued"


def test_g0_no_nan_inf():
    """g_0 must not contain NaN or Inf."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    g0, _, _ = enc(x)

    assert torch.isfinite(g0).all(), "g_0 contains NaN or Inf"


# ================================================================================
# 5. STOCHASTICITY
# ================================================================================
def test_g0_stochastic_in_train_mode():
    """Two forward passes in train mode must produce different g_0 samples."""
    from models.encoder import Encoder

    enc = Encoder()
    enc.train()
    x = torch.randn(T_DATA, N_ROIS)

    g0_1, _, _ = enc(x)
    g0_2, _, _ = enc(x)

    assert not torch.allclose(g0_1, g0_2), \
        "g_0 is identical across two train-mode calls — sampling may be broken"


def test_g0_deterministic_in_eval_mode():
    """In eval mode with fixed seed, g_0 must be deterministic."""
    from models.encoder import Encoder

    enc = Encoder()
    enc.eval()
    x = torch.randn(T_DATA, N_ROIS)

    torch.manual_seed(0)
    g0_1, _, _ = enc(x)
    torch.manual_seed(0)
    g0_2, _, _ = enc(x)

    assert torch.allclose(g0_1, g0_2), \
        "g_0 is not deterministic in eval mode with fixed seed"


# ================================================================================
# 6. PERMUTATION EQUIVARIANCE OF SPATIO-ENCODER NETWORK
# ================================================================================
def test_phi_permutation_equivariant():
    """Paper's actual claim: Phi(sigma . Z0) = sigma . Phi(Z0).
    Permute Z0 ROWS (not x) and check the transformer+proj output permutes."""
    enc = Encoder(); enc.eval()
    torch.manual_seed(0)
    z0 = enc.encode_spatial(torch.randn(T_DATA, N_ROIS))
    perm = torch.randperm(N_ROIS)
    mu, logvar = enc.node_params(z0.unsqueeze(0))
    mu_p, logvar_p = enc.node_params(z0[perm].unsqueeze(0))
    assert torch.allclose(mu[0][perm], mu_p[0], atol=1e-5)
    assert torch.allclose(logvar[0][perm], logvar_p[0], atol=1e-5)


# ================================================================================
# 7. PRIOR CHECK
# ================================================================================
def test_g0_prior_mean_near_zero():
    """
    Untrained encoder g_0 samples should have mean close to 0,
    consistent with the prior g_0 ~ N(0, epsilon*I).
    """
    from models.encoder import Encoder

    torch.manual_seed(42)
    enc = Encoder()
    enc.train()
    x = torch.randn(T_DATA, N_ROIS)

    samples = torch.stack([enc(x)[0] for _ in range(100)])
    mean = samples.mean(dim=0)

    assert mean.abs().mean().item() < 1.0, \
        f"g_0 mean is far from 0: {mean.abs().mean().item():.4f}"


# ================================================================================
# 8. GRADIENT FLOW
# ================================================================================
def test_gradients_flow_through_encoder():
    """All encoder parameters must receive gradients on a forward+backward pass."""
    from models.encoder import Encoder

    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    g0, _, _ = enc(x)
    g0.sum().backward()

    for name, param in enc.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_H_divisible_by_nhead():
    assert H % NHEAD == 0, f"H={H} must be divisible by NHEAD={NHEAD}"
 
 
def test_latent_dim_consistency():
    enc = Encoder()
    assert enc.m == N_ROIS * H == M
    assert enc(torch.randn(T_DATA, N_ROIS))[0].shape == (M,)


def test_reparam_matches_distribution():
    """Samples must actually be ~ N(mu, exp(0.5 logvar))."""
    torch.manual_seed(0)
    enc = Encoder(); enc.train()
    x = torch.randn(T_DATA, N_ROIS)
    mu, logvar = enc.encode_distribution(x)
    samples = torch.stack([enc(x)[0] for _ in range(8000)])
    assert (samples.mean(0) - mu).abs().max() < 0.1
    assert (samples.std(0) - torch.exp(0.5 * logvar)).abs().max() < 0.1
 
 
def test_eval_returns_posterior_mean():
    enc = Encoder(); enc.eval()
    x = torch.randn(T_DATA, N_ROIS)
    mu, _ = enc.encode_distribution(x)
    assert torch.allclose(enc(x)[0], mu)
 
 
def test_batch_support():
    enc = Encoder()
    g0, _, _ = enc(torch.randn(8, T_DATA, N_ROIS))
    assert g0.shape == (8, M)
 
 
def test_logvar_is_clamped():
    enc = Encoder(logvar_clamp=(-4.0, 4.0))
    _, logvar = enc.encode_distribution(torch.randn(T_DATA, N_ROIS))
    assert logvar.min() >= -4.0 - 1e-5 and logvar.max() <= 4.0 + 1e-5



# ================================================================================
# forward TESTS
# ================================================================================
def test_forward_returns_three_values():
    """forward must return g_0, mu, logvar."""
    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    result = enc.forward(x)
    assert len(result) == 3, f"Expected 3 return values, got {len(result)}"


def test_forward_shapes():
    """g_0, mu, logvar must all be shape (M,)."""
    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    g_0, mu, logvar = enc.forward(x)
    assert g_0.shape    == (M,), f"g_0 shape {g_0.shape}, expected ({M},)"
    assert mu.shape     == (M,), f"mu shape {mu.shape}, expected ({M},)"
    assert logvar.shape == (M,), f"logvar shape {logvar.shape}, expected ({M},)"


def test_forward_no_nan_inf():
    """No NaN or Inf in any output."""
    enc = Encoder()
    x = torch.randn(T_DATA, N_ROIS)
    g_0, mu, logvar = enc.forward(x)
    assert torch.isfinite(g_0).all(),    "g_0 contains NaN or Inf"
    assert torch.isfinite(mu).all(),     "mu contains NaN or Inf"
    assert torch.isfinite(logvar).all(), "logvar contains NaN or Inf"


def test_forward_stochastic_in_train():
    """Two calls in train mode must produce different g_0 but same mu and logvar."""
    enc = Encoder()
    enc.train()
    x = torch.randn(T_DATA, N_ROIS)
    g0_1, mu_1, logvar_1 = enc.forward(x)
    g0_2, mu_2, logvar_2 = enc.forward(x)
    assert not torch.allclose(g0_1, g0_2), "g_0 should differ between train calls"
    assert torch.allclose(mu_1, mu_2),     "mu should be identical between calls"
    assert torch.allclose(logvar_1, logvar_2), "logvar should be identical between calls"


def test_forward_deterministic_in_eval():
    """In eval mode, g_0 must equal mu."""
    enc = Encoder()
    enc.eval()
    x = torch.randn(T_DATA, N_ROIS)
    g_0, mu, _ = enc.forward(x)
    assert torch.allclose(g_0, mu), "g_0 must equal mu in eval mode"


def test_forward_consistent_with_encode_distribution():
    """mu and logvar must match encode_distribution output."""
    enc = Encoder()
    enc.eval()
    x = torch.randn(T_DATA, N_ROIS)
    _, mu_sample, logvar_sample = enc.forward(x)
    mu_dist, logvar_dist = enc.encode_distribution(x)
    assert torch.allclose(mu_sample, mu_dist),         "mu mismatch with encode_distribution"
    assert torch.allclose(logvar_sample, logvar_dist), "logvar mismatch with encode_distribution"



def test_g0_prior_variance_near_epsilon():
    """
    Untrained encoder g_0 samples should have std close to sqrt(epsilon),
    consistent with the prior g_0 ~ N(0, epsilon*I).
    """
    from config import EPSILON
    torch.manual_seed(42)
    enc = Encoder()
    enc.train()
    x = torch.randn(T_DATA, N_ROIS)

    samples = torch.stack([enc(x)[0] for _ in range(500)])
    std = samples.std(dim=0)
    expected_std = EPSILON ** 0.5  # 1.0 with default config

    assert (std.mean() - expected_std).abs() < 0.5, \
        f"g_0 std {std.mean():.4f} far from expected {expected_std:.4f}"