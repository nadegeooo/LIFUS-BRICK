# tests/test_control.py

import torch
from config import M, N_ROIS, T as T_DATA
from models.control import ControlModule


# ================================================================================
# 1. C MATRIX TESTS
# ================================================================================
def test_C_is_diagonal():
    """C must be diagonal — off-diagonal elements must be zero."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    C, _, _, _, _ = ctrl(x)

    off_diag = C - torch.diag(torch.diag(C))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6), \
        "C has non-zero off-diagonal elements"


def test_C_diag_in_range():
    """diag(C) values must be in (-1, 1) from tanh."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    C, _, _, _, _ = ctrl(x)

    diag = torch.diag(C)
    assert (diag > -1.0).all() and (diag < 1.0).all(), \
        f"diag(C) values outside (-1, 1): min={diag.min().item():.4f}, max={diag.max().item():.4f}"


def test_C_shape():
    """C must be shape (M, M)."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    C, _, _, _, _ = ctrl(x)

    assert C.shape == (M, M), f"Expected ({M}, {M}), got {C.shape}"


def test_C_no_nan_inf():
    """C must not contain NaN or Inf."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    C, _, _, _, _ = ctrl(x)

    assert torch.isfinite(C).all(), "C contains NaN or Inf"


def test_C_subject_specific():
    """Different inputs must produce different C matrices."""
    from models.control import ControlModule

    ctrl = ControlModule()
    ctrl.eval()

    x1 = torch.randn(T_DATA, N_ROIS)
    x2 = torch.randn(T_DATA, N_ROIS)

    C1, _, _, _, _ = ctrl(x1)
    C2, _, _, _, _ = ctrl(x2)

    assert not torch.allclose(C1, C2), \
        "C is identical for different inputs — not subject-specific"


# ================================================================================
# 2. CONTROL INPUT u TESTS
# ================================================================================
def test_u_shape():
    """Control input u must be shape (T, M)."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, u, _, _, _ = ctrl(x)

    assert u.shape == (T_DATA, M), \
        f"Expected ({T_DATA}, {M}), got {u.shape}"


def test_u_no_nan_inf():
    """u must not contain NaN or Inf."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, u, _, _, _ = ctrl(x)

    assert torch.isfinite(u).all(), "u contains NaN or Inf"


def test_u_stochastic_in_train_mode():
    """Two forward passes in train mode must produce different u samples."""
    from models.control import ControlModule

    ctrl = ControlModule()
    ctrl.train()
    x = torch.randn(T_DATA, N_ROIS)

    _, u1, _, _, _ = ctrl(x)
    _, u2, _, _, _ = ctrl(x)

    assert not torch.allclose(u1, u2), \
        "u is identical across two train-mode calls — sampling may be broken"


def test_u_deterministic_in_eval_mode():
    """In eval mode, u must be deterministic (returns mu_u)."""
    from models.control import ControlModule

    ctrl = ControlModule()
    ctrl.eval()
    x = torch.randn(T_DATA, N_ROIS)

    torch.manual_seed(0)
    _, u1, _, _, _ = ctrl(x)
    torch.manual_seed(0)
    _, u2, _, _, _ = ctrl(x)

    assert torch.allclose(u1, u2), \
        "u is not deterministic in eval mode"


# ================================================================================
# 3. CLASSIFIER OUTPUT TESTS
# ================================================================================
def test_classifier_output_shape():
    """Classifier output s_hat must be shape (num_classes,)."""
    from models.control import ControlModule

    num_classes = 2
    ctrl = ControlModule(num_classes=num_classes)
    x = torch.randn(T_DATA, N_ROIS)
    _, _, s_hat, _, _ = ctrl(x)

    assert s_hat.shape == (num_classes,), \
        f"Expected ({num_classes},), got {s_hat.shape}"


def test_classifier_no_nan_inf():
    """Classifier output must not contain NaN or Inf."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, _, s_hat, _, _ = ctrl(x)

    assert torch.isfinite(s_hat).all(), "s_hat contains NaN or Inf"


# ================================================================================
# 4. EMBEDDING TESTS
# ================================================================================
def test_embedding_shape():
    """Embedding E must be shape (M, T)."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    E = ctrl.encode(x)

    assert E.shape == (M, T_DATA), \
        f"Expected ({M}, {T_DATA}), got {E.shape}"


def test_embedding_no_nan_inf():
    """Embedding E must not contain NaN or Inf."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    E = ctrl.encode(x)

    assert torch.isfinite(E).all(), "E contains NaN or Inf"


# ================================================================================
# 5. GRADIENT FLOW
# ================================================================================
def test_gradients_flow_through_control():
    """All control module parameters must receive gradients."""
    from models.control import ControlModule

    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    C, u, s_hat, _, _ = ctrl(x)

    loss = C.sum() + u.sum() + s_hat.sum()
    loss.backward()

    for name, param in ctrl.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


# ================================================================================
# 6. MU_U AND LOGVAR_U TESTS
# ================================================================================
def test_mu_u_shape():
    """mu_u must be shape (T, M)."""
    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, _, _, mu_u, _ = ctrl(x)
    assert mu_u.shape == (T_DATA, M), \
        f"Expected ({T_DATA}, {M}), got {mu_u.shape}"


def test_logvar_u_shape():
    """logvar_u must be shape (T, M)."""
    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, _, _, _, logvar_u = ctrl(x)
    assert logvar_u.shape == (T_DATA, M), \
        f"Expected ({T_DATA}, {M}), got {logvar_u.shape}"


def test_mu_u_logvar_u_same_shape():
    """mu_u and logvar_u must have the same shape."""
    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, _, _, mu_u, logvar_u = ctrl(x)
    assert mu_u.shape == logvar_u.shape, \
        f"mu_u {mu_u.shape} != logvar_u {logvar_u.shape}"


def test_mu_u_no_nan_inf():
    """mu_u must not contain NaN or Inf."""
    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, _, _, mu_u, _ = ctrl(x)
    assert torch.isfinite(mu_u).all(), "mu_u contains NaN or Inf"


def test_logvar_u_no_nan_inf():
    """logvar_u must not contain NaN or Inf."""
    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, _, _, _, logvar_u = ctrl(x)
    assert torch.isfinite(logvar_u).all(), "logvar_u contains NaN or Inf"


def test_logvar_u_clamped():
    """logvar_u must be within clamp range (-10, 10)."""
    ctrl = ControlModule()
    x = torch.randn(T_DATA, N_ROIS)
    _, _, _, _, logvar_u = ctrl(x)
    assert (logvar_u >= -10.0 - 1e-5).all() and (logvar_u <= 10.0 + 1e-5).all(), \
        f"logvar_u outside clamp range: min={logvar_u.min().item():.4f}, max={logvar_u.max().item():.4f}"


def test_mu_u_deterministic_in_eval():
    """mu_u must be deterministic in eval mode."""
    ctrl = ControlModule()
    ctrl.eval()
    x = torch.randn(T_DATA, N_ROIS)
    _, _, _, mu_u_1, _ = ctrl(x)
    _, _, _, mu_u_2, _ = ctrl(x)
    assert torch.allclose(mu_u_1, mu_u_2), "mu_u is not deterministic in eval mode"


def test_u_equals_mu_u_in_eval():
    """In eval mode, u must equal mu_u (no sampling)."""
    ctrl = ControlModule()
    ctrl.eval()
    x = torch.randn(T_DATA, N_ROIS)
    _, u, _, mu_u, _ = ctrl(x)
    assert torch.allclose(u, mu_u), "u != mu_u in eval mode — sampling should be disabled"


# ================================================================================
# 7. BATCH SUPPORT
# ================================================================================
def test_batch_support():
    """Control module must handle batched input (B, T, N)."""
    from models.control import ControlModule

    B = 4
    ctrl = ControlModule()
    x = torch.randn(B, T_DATA, N_ROIS)
    C, u, s_hat, _, _ = ctrl(x)

    assert C.shape == (B, M, M), f"Expected ({B}, {M}, {M}), got {C.shape}"
    assert u.shape == (B, T_DATA, M), f"Expected ({B}, {T_DATA}, {M}), got {u.shape}"
    assert s_hat.shape == (B, 2), f"Expected ({B}, 2), got {s_hat.shape}"
    