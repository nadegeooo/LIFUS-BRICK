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
    x     = torch.randn(T_DATA, N_ROIS)
    label = torch.tensor(0)
    return x, label


# ================================================================================
# 1. FORWARD PASS OUTPUT SHAPES
# ================================================================================
def test_x_recon_shape(model, dummy_input):
    """x_recon must be (T, N) matching input shape."""
    x, label = dummy_input
    out = model(x, label)
    assert out["x_recon"].shape == (T_DATA, N_ROIS), \
        f"Expected ({T_DATA}, {N_ROIS}), got {out['x_recon'].shape}"


def test_g_trajectory_shape(model, dummy_input):
    """g_trajectory must be (T, M)."""
    x, label = dummy_input
    out = model(x, label)
    assert out["g_trajectory"].shape == (T_DATA, M), \
        f"Expected ({T_DATA}, {M}), got {out['g_trajectory'].shape}"


def test_C_shape(model, dummy_input):
    """C must be (M, M)."""
    x, label = dummy_input
    out = model(x, label)
    assert out["C"].shape == (M, M), \
        f"Expected ({M}, {M}), got {out['C'].shape}"


def test_s_hat_shape(model, dummy_input):
    """s_hat must be (num_classes,)."""
    x, label = dummy_input
    out = model(x, label)
    assert out["s_hat"].shape == (NUM_CLASSES,), \
        f"Expected ({NUM_CLASSES},), got {out['s_hat'].shape}"


# ================================================================================
# 2. LOSS DICTIONARY
# ================================================================================
def test_loss_keys(model, dummy_input):
    """Loss dict must contain all required keys."""
    x, label = dummy_input
    out = model(x, label)
    required = {"loss_total", "loss_recon", "loss_kl_g0", "loss_kl_u", "loss_cls"}
    assert required == set(out["losses"].keys()), \
        f"Missing keys: {required - set(out['losses'].keys())}"


def test_losses_are_scalar(model, dummy_input):
    """All losses must be scalar tensors."""
    x, label = dummy_input
    out = model(x, label)
    for name, loss in out["losses"].items():
        assert loss.shape == torch.Size([]), \
            f"{name} is not scalar: shape={loss.shape}"


def test_losses_are_finite(model, dummy_input):
    """All losses must be finite."""
    x, label = dummy_input
    out = model(x, label)
    for name, loss in out["losses"].items():
        assert torch.isfinite(loss), f"{name} is not finite: {loss.item()}"


# ================================================================================
# 3. OUTPUT VALUE CHECKS
# ================================================================================
def test_x_recon_no_nan_inf(model, dummy_input):
    """x_recon must not contain NaN or Inf."""
    x, label = dummy_input
    out = model(x, label)
    assert torch.isfinite(out["x_recon"]).all(), "x_recon contains NaN or Inf"


def test_g_trajectory_no_nan_inf(model, dummy_input):
    """g_trajectory must not contain NaN or Inf."""
    x, label = dummy_input
    out = model(x, label)
    assert torch.isfinite(out["g_trajectory"]).all(), \
        "g_trajectory contains NaN or Inf"


def test_C_is_diagonal(model, dummy_input):
    """C must be diagonal."""
    x, label = dummy_input
    out = model(x, label)
    C = out["C"]
    off_diag = C - torch.diag(torch.diag(C))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-6), \
        "C has non-zero off-diagonal elements"


def test_C_diag_in_range(model, dummy_input):
    """diag(C) values must be in (-1, 1) from tanh."""
    x, label = dummy_input
    out = model(x, label)
    diag = torch.diag(out["C"])
    assert (diag > -1.0).all() and (diag < 1.0).all(), \
        f"diag(C) outside (-1, 1): min={diag.min().item():.4f}, max={diag.max().item():.4f}"


# ================================================================================
# 4. LOSS DECREASES AFTER ONE GRADIENT STEP
# ================================================================================
def test_loss_decreases_after_gradient_step():
    """loss_total must decrease after one gradient step."""
    torch.manual_seed(0)
    model     = BRICK()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x         = torch.randn(T_DATA, N_ROIS)
    label     = torch.tensor(0)

    # First forward pass
    out        = model(x, label)
    loss_before = out["losses"]["loss_total"].item()

    # Gradient step
    optimizer.zero_grad()
    out["losses"]["loss_total"].backward()
    optimizer.step()

    # Second forward pass
    out        = model(x, label)
    loss_after = out["losses"]["loss_total"].item()

    assert loss_after < loss_before, \
        f"Loss did not decrease: before={loss_before:.4f}, after={loss_after:.4f}"


# ================================================================================
# 5. ABLATION FLAGS
# ================================================================================
def test_ablation_no_control():
    """use_control=False must run without error."""
    torch.manual_seed(0)
    model = BRICK(use_control=False)
    x     = torch.randn(T_DATA, N_ROIS)
    out   = model(x)
    assert out["x_recon"].shape == (T_DATA, N_ROIS)
    assert torch.isfinite(out["losses"]["loss_total"])


def test_ablation_no_ic():
    """use_ic=False must run without error."""
    torch.manual_seed(0)
    model = BRICK(use_ic=False)
    x     = torch.randn(T_DATA, N_ROIS)
    out   = model(x)
    assert out["x_recon"].shape == (T_DATA, N_ROIS)
    assert torch.isfinite(out["losses"]["loss_total"])


def test_ablation_no_control_no_ic():
    """use_control=False and use_ic=False must run without error."""
    torch.manual_seed(0)
    model = BRICK(use_control=False, use_ic=False)
    x     = torch.randn(T_DATA, N_ROIS)
    out   = model(x)
    assert out["x_recon"].shape == (T_DATA, N_ROIS)
    assert torch.isfinite(out["losses"]["loss_total"])


# ================================================================================
# 6. CONFIG ASSERTION
# ================================================================================
def test_config_mismatch_raises():
    """BRICK must raise AssertionError if m != n_rois * h."""
    with pytest.raises(AssertionError):
        BRICK(n_rois=24, h=8, m=100)  # 24*8=192 != 100


# ================================================================================
# 7. GRADIENT FLOW
# ================================================================================
def test_gradients_flow_through_brick():
    """All BRICK parameters must receive gradients on a forward+backward pass."""
    torch.manual_seed(0)
    model = BRICK()
    x     = torch.randn(T_DATA, N_ROIS)
    label = torch.tensor(0)
    out   = model(x, label)
    out["losses"]["loss_total"].backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"