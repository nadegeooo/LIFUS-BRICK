import torch
import torch.nn as nn
from models.koopman_utils import compute_pearson
from config import N_ROIS, H, M, MLP_HIDDEN, NHEAD, NUM_LAYERS


class Encoder(nn.Module):
    def __init__(self, n_rois=N_ROIS, h=H, m=M, mlp_hidden=MLP_HIDDEN,
                 nhead=NHEAD, num_layers=NUM_LAYERS, dropout=0.1, logvar_clamp=(-10.0, 10.0)):
        super().__init__()
        # FIX: guard the d_model % nhead == 0 requirement up front
        assert h % nhead == 0, f"H={h} must be divisible by NHEAD={nhead}"
        self.n_rois = n_rois
        self.h = h
        self.m = m
        self.logvar_clamp = logvar_clamp

        self.row_mlp = nn.Sequential(
            nn.Linear(n_rois, mlp_hidden), 
            nn.ReLU(), 
            nn.Linear(mlp_hidden, h),
        )

        def make_head():
            layer = nn.TransformerEncoderLayer(
                d_model=h, nhead=nhead, batch_first=True, dim_feedforward=h * 4, dropout=dropout
            )
            return nn.TransformerEncoder(layer, num_layers=num_layers)

        self.mu_head, self.logvar_head = make_head(), make_head()
        self.mu_proj, self.logvar_proj = nn.Linear(h, h), nn.Linear(h, h)

    def encode_spatial(self, x):
        # (T, N) or (B, T, N) -> (N, H) or (B, N, H). row_mlp shares weights
        # across rows, so output row i tracks region i.
        return self.row_mlp(compute_pearson(x))

    def node_params(self, z0):
        # z0: (..., N, H) -> per-node mu, logvar of shape (..., N, H).
        # Operates token-wise; mu_head/logvar_head are equivariant over the N
        # tokens, so this is the layer the paper's Phi-equivariance claim is about.
        mu = self.mu_proj(self.mu_head(z0))
        logvar = self.logvar_proj(self.logvar_head(z0))
        LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0
        logvar = LOGVAR_MIN + 0.5 * (LOGVAR_MAX - LOGVAR_MIN) * (torch.tanh(logvar) + 1.0)
        return mu, logvar

    def encode_distribution(self, x):
        z0 = self.encode_spatial(x)               # (..., N, H)
        batched = (z0.dim() == 3)
        if not batched:
            z0 = z0.unsqueeze(0)                  # FIX: real (B,...) handling
        mu, logvar = self.node_params(z0)         # (B, N, H)
        # row-major flatten: region i -> block [i*H:(i+1)*H], i.e. N blocks of H.
        # This is the layout the control-matrix->region interpretation relies on;
        # keep it consistent with the decoder / W_x indexing.
        mu = mu.reshape(mu.shape[0], self.m)
        logvar = logvar.reshape(logvar.shape[0], self.m)
        if not batched:
            mu, logvar = mu.squeeze(0), logvar.squeeze(0)
        return mu, logvar

    def forward(self, x: torch.Tensor):
        """
        Full encoder forward pass.

        Returns g_0 sample, mu, and logvar from the approximate posterior.

        In train mode: g_0 = mu + eps * exp(0.5 * logvar),  eps ~ N(0, I)
        In eval mode:  g_0 = mu (deterministic)

        Args:
            x (torch.Tensor): BOLD timeseries, shape (T, N)

        Returns:
            g_0    (torch.Tensor): Sample, shape (M,)
            mu     (torch.Tensor): Posterior mean, shape (M,)
            logvar (torch.Tensor): Posterior log variance, shape (M,)
        """
        mu, logvar = self.encode_distribution(x)

        if self.training:
            eps = torch.randn_like(mu)
            g_0 = mu + eps * torch.exp(0.5 * logvar)
        else:
            g_0 = mu

        return g_0, mu, logvar