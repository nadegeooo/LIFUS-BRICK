"""
================================================================================
Control Module
================================================================================

Description:
    Implements the control module component of BRICK (Section III-C).

    The control module extends the autonomous Koopman system to a closed-loop
    feedback control system:

        g_{t+1} = K g_t + C u_{t+1}

    where C is a subject-specific diagonal control matrix and u_{1:T} are
    external control inputs generated from the BOLD observations.

    The module has five outputs:
        1. C  — diagonal control matrix, shape (M, M), diag values in (-1, 1)
        2. u  — control inputs, shape (T, M), sampled via reparameterization
        3. s_hat — task state prediction (pre/post), shape (num_classes,)
        4.
        5.

    Architecture:
        - TransformerEncoder: x_{1:T} (T, N) -> E (M, T)
        - Average pool E over time -> e (M,)
        - diag(C) = tanh(W e + b)        [diagonal control matrix]
        - MLP_mu(E)    -> mu_u  (T, M)   [control input mean]
        - MLP_sig(E)   -> logvar_u (T, M) [control input log variance]
        - u_t ~ N(mu_u_t, exp(0.5 * logvar_u_t)) via reparameterization
        - Linear(e) -> s_hat (num_classes,) [task classifier]

References:
    - Zhou et al. 2025 (BRICK paper), Section III-C
"""

import torch
import torch.nn as nn
from config import N_ROIS, M, NHEAD, NUM_LAYERS


class ControlModule(nn.Module):
    """
    Control module generating subject-specific C, control inputs u, and
    task state predictions s_hat from BOLD observations.

    Args:
        n_rois      (int): Number of ROIs (N). Default from config.
        m           (int): Latent dimension (M). Default from config.
        nhead       (int): Attention heads in TransformerEncoder.
        num_layers  (int): Number of TransformerEncoder layers.
        num_classes (int): Number of task states to classify. Default 2 (pre/post).
    """

    def __init__(
        self,
        n_rois:      int   = N_ROIS,
        m:           int   = M,
        nhead:       int   = NHEAD,
        num_layers:  int   = NUM_LAYERS,
        num_classes: int   = 2,
    ):
        super().__init__()

        self.n_rois      = n_rois
        self.m           = m
        self.num_classes = num_classes

        # --- Transformer Encoder: x (T, N) -> E (T, M) ---
        # Projects N -> M first, then applies transformer over T tokens
        self.input_proj = nn.Linear(n_rois, m)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=m, nhead=nhead, batch_first=True, dim_feedforward=m * 4, dropout=0.1         # 4 is feedforward multiplier, common default in transformer architectures (see Attention Is All You Need paper)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Control matrix generation: e (M,) -> diag(C) (M,) ---
        self.C_gate = nn.Linear(m, m)

        # --- Control input generation: E (T, M) -> mu_u, logvar_u (T, M) ---
        self.mlp_mu  = nn.Sequential(nn.Linear(m, m * 2), nn.ReLU(), nn.Linear(m * 2, m))
        self.mlp_sig = nn.Sequential(nn.Linear(m, m * 2), nn.ReLU(), nn.Linear(m * 2, m))

        # --- Task state classifier: e (M,) -> s_hat (num_classes,) ---
        self.classifier = nn.Linear(m, num_classes)

        # Initialize logvar head near zero -> logvar ≈ -2 at init, sigma ≈ 0.37
        nn.init.zeros_(self.mlp_sig[-1].weight)
        nn.init.zeros_(self.mlp_sig[-1].bias) 

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode BOLD observations into embedding E.

        Args:
            x (torch.Tensor): BOLD timeseries, shape (T, N) or (B, T, N)

        Returns:
            torch.Tensor: Embedding E, shape (M, T) or (B, M, T)
        """
        batched = (x.dim() == 3)
        if not batched:
            x = x.unsqueeze(0)                     # (1, T, N)

        x_proj = self.input_proj(x)                # (B, T, M)
        E = self.transformer(x_proj)               # (B, T, M)
        E = E.permute(0, 2, 1)                     # (B, M, T)

        if not batched:
            E = E.squeeze(0)                        # (M, T)

        return E

    def forward(self, x: torch.Tensor):
        """
        Full control module forward pass.

        Args:
            x (torch.Tensor): BOLD timeseries, shape (T, N) or (B, T, N)

        Returns:
            C     (torch.Tensor): Diagonal control matrix, shape (M, M) or (B, M, M)
            u     (torch.Tensor): Control inputs, shape (T, M) or (B, T, M)
            s_hat (torch.Tensor): Task state logits, shape (num_classes,) or (B, num_classes)
        """
        batched = (x.dim() == 3)

        E = self.encode(x)                          # (M, T) or (B, M, T)

        # Always work in batched form internally
        if not batched:
            E = E.unsqueeze(0)  # (1, M, T)

        # Average pool over time -> e (B, M)
        e = E.mean(dim=-1)                          # (B, M)

        # Control matrix: diag(C) = tanh(W e + b)
        diag_C = torch.tanh(self.C_gate(e))         # (B, M)
        C = torch.diag_embed(diag_C)               # (B, M, M)

        # Control inputs from E: (B, M, T) -> (B, T, M)
        E_t      = E.permute(0, 2, 1)              # (B, T, M)
        mu_u     = self.mlp_mu(E_t)                # (B, T, M)
        LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0    # sigma in [0.05, 2.7]
        logvar_u = self.mlp_sig(E_t)               # (B, T, M)
        logvar_u = LOGVAR_MIN + 0.5 * (LOGVAR_MAX - LOGVAR_MIN) * (torch.tanh(logvar_u) + 1.0)

        if self.training:
            eps = torch.randn_like(mu_u)
            u = mu_u + eps * torch.exp(0.5 * logvar_u)
        else:
            u = mu_u

        s_hat = self.classifier(e)                 # (B, num_classes)

        # Squeeze back if input was unbatched
        if not batched:
            C     = C.squeeze(0)       # (M, M)
            u     = u.squeeze(0)       # (T, M)
            mu_u  = mu_u.squeeze(0)    # (T, M)
            logvar_u = logvar_u.squeeze(0)  # (T, M)
            s_hat = s_hat.squeeze(0)   # (num_classes,)

        return C, u, s_hat, mu_u, logvar_u