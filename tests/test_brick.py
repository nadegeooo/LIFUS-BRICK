# tests/test_brick.py

import pytest
import torch
from config import M, N_ROIS, H, T as T_DATA, NUM_CLASSES
from models.brick import BRICK


# ================================================================================
# FIXTURES
# ================================================================================
@pytest.fixture
def model():
    return BRICK()


@pytest.fixture
def dummy_input():
    torch.manual_seed(0)
    x = torch.randn(T_DATA, N_ROIS)
    label = torch.tensor(0)
    return x, label


# ================================================================================
# 1. FORWARD PASS OUTPUT SHAPES / DTYPES
# ================================================================================
def test_x_recon_shape(model, dummy_input):
    """x_recon must be (T, N) matching input shape."""
    x, label = dummy_input
    out = model(x, label)
    assert out["x_recon"].shape == (T_DATA, N_ROIS), \
        f"Expected ({T_DATA}, {N_ROIS}), got {out['x_recon'].shape}"


def test_x_recon_is_real(model, dummy_input):
    """
    x_recon must be REAL. The reconstruction is real(W_bar_x @ g_bar); if the
    .real were ever dropped, x_recon would go complex and MSE against real x
    would silently misbehave. isfinite alone would not catch that, so assert
    dtype explicitly.
    """
    x, label = dummy_input
    out = model(x, label)
    assert not out["x_recon"].is_complex(), "x_recon must be real, got complex"


def test_g_trajectory_shape(model, dummy_input):
    """g_trajectory must be (T, M)."""
    x, label = dummy_input
    out = model(x, label)
    assert out["g_trajectory"].shape == (T_DATA, M), \
        f"Expected ({T_DATA}, {M}), got {out['g_trajectory'].shape}"


def test_g_trajectory_is_complex(model, dummy_input):
    """
    Post-refactor contract: g_trajectory is the EIGENSPACE trajectory g_bar and
    is complex. There is no measurement-space real trajectory anymore (that
    required P, which the inversion-free refactor removed). Downstream code must
    not assume real, region-structured values here. ELBO/recon use x_recon and
    C, not this; it is exposed for inspection only.
    """
    x, label = dummy_input
    out = model(x, label)
    assert out["g_trajectory"].is_complex(), \
        "g_trajectory should be complex (eigenspace) post-refactor"


def test_C_shape(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    assert out["C"].shape == (M, M), f"Expected ({M}, {M}), got {out['C'].shape}"


def test_s_hat_shape(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    assert out["s_hat"].shape == (NUM_CLASSES,), \
        f"Expected ({NUM_CLASSES},), got {out['s_hat'].shape}"


# ================================================================================
# 2. LOSS DICTIONARY
# ================================================================================
def test_loss_keys(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    required = {"loss_total", "loss_recon", "loss_kl_g0", "loss_kl_u", "loss_cls"}
    assert required == set(out["losses"].keys())


def test_losses_are_scalar(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    for name, loss in out["losses"].items():
        assert loss.shape == torch.Size([]), f"{name} not scalar: {loss.shape}"


def test_losses_are_finite(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    for name, loss in out["losses"].items():
        assert torch.isfinite(loss), f"{name} not finite: {loss.item()}"


# ================================================================================
# 3. OUTPUT VALUE CHECKS
# ================================================================================
def test_x_recon_no_nan_inf(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    assert torch.isfinite(out["x_recon"]).all()


def test_g_trajectory_no_nan_inf(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    assert torch.isfinite(out["g_trajectory"].abs()).all()


def test_C_is_diagonal(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    C = out["C"]
    off_diag = C - torch.diag(torch.diag(C))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6)


def test_C_diag_in_range(model, dummy_input):
    x, label = dummy_input
    out = model(x, label)
    diag = torch.diag(out["C"])
    assert (diag > -1.0).all() and (diag < 1.0).all()


# ================================================================================
# 4. LOSS DECREASES AFTER ONE GRADIENT STEP (deterministic)
# ================================================================================
def test_loss_decreases_after_gradient_step():
    """
    loss_total must decrease after one gradient step. Seed identically before
    BOTH forward passes so the reparameterization draws (encoder g_0 and control
    u) are the same in both, isolating the effect of the parameter update from
    VAE sampling noise. Without this, sampling variance alone can make the loss
    rise and flake the test.
    """
    torch.manual_seed(0)
    model = BRICK()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(T_DATA, N_ROIS)
    label = torch.tensor(0)

    torch.manual_seed(123)
    loss_before = model(x, label)["losses"]["loss_total"]

    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    loss_before_val = loss_before.item()

    torch.manual_seed(123)
    loss_after_val = model(x, label)["losses"]["loss_total"].item()

    assert loss_after_val < loss_before_val, \
        f"Loss did not decrease: before={loss_before_val:.4f}, after={loss_after_val:.4f}"


# ================================================================================
# 5. ABLATION FLAGS
# ================================================================================
def test_ablation_no_control():
    torch.manual_seed(0)
    model = BRICK(use_control=False)
    out = model(torch.randn(T_DATA, N_ROIS))
    assert out["x_recon"].shape == (T_DATA, N_ROIS)
    assert torch.isfinite(out["losses"]["loss_total"])


def test_ablation_no_ic():
    torch.manual_seed(0)
    model = BRICK(use_ic=False)
    out = model(torch.randn(T_DATA, N_ROIS))
    assert out["x_recon"].shape == (T_DATA, N_ROIS)
    assert torch.isfinite(out["losses"]["loss_total"])


def test_ablation_no_control_no_ic():
    torch.manual_seed(0)
    model = BRICK(use_control=False, use_ic=False)
    out = model(torch.randn(T_DATA, N_ROIS))
    assert out["x_recon"].shape == (T_DATA, N_ROIS)
    assert torch.isfinite(out["losses"]["loss_total"])


# ================================================================================
# 6. CONFIG ASSERTION
# ================================================================================
def test_config_mismatch_raises():
    with pytest.raises(AssertionError):
        BRICK(n_rois=24, h=8, m=100)  # 24*8=192 != 100


# ================================================================================
# 7. GRADIENT FLOW
# ================================================================================
def test_gradients_flow_through_brick():
    """All BRICK parameters must receive finite gradients."""
    torch.manual_seed(0)
    model = BRICK()
    x = torch.randn(T_DATA, N_ROIS)
    label = torch.tensor(0)
    model(x, label)["losses"]["loss_total"].backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_P_inv_receives_gradient():
    """
    Carved out from the omnibus test so a failure names the culprit. P_inv is
    the inversion-free input map; it reaches the loss only through the complex
    scan and the real(W_bar_x @ g_bar) boundary. A detach or a dropped .real
    path would silently freeze it, making the refactor decorative. Assert it
    gets a finite, NONZERO gradient.
    """
    torch.manual_seed(0)
    model = BRICK()
    x = torch.randn(T_DATA, N_ROIS)
    label = torch.tensor(0)
    model(x, label)["losses"]["loss_total"].backward()
    g = model.P_inv.grad
    assert g is not None, "P_inv received no gradient"
    assert torch.isfinite(g).all(), "P_inv gradient is non-finite"
    assert g.abs().sum() > 0, "P_inv gradient is all zero — input map is frozen"


def test_W_bar_x_receives_gradient():
    """W_bar_x (complex output map) must also get a finite, nonzero gradient."""
    torch.manual_seed(0)
    model = BRICK()
    x = torch.randn(T_DATA, N_ROIS)
    label = torch.tensor(0)
    model(x, label)["losses"]["loss_total"].backward()
    g = model.W_bar_x.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0