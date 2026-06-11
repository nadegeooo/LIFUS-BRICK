"""Keep all configuration parameters in one place for easy access and modification."""

N_ROIS = 24
T = 240        # Number of timepoints
H = 2          # MLP hidden multiplier
M = N_ROIS * H # Koopman latent dimension (48)

# Koopman eigenvalue initialization -> R_MIN and R_MAX control the minimum and maximum magnitude of eigenvalues at initialization
# Values close to 1.0 give slow dynamics, which is appropriate for fMRI data. Setting R_MAX < 1.0 ensures stability of the Koopman operator.
# Low R_MIN results in short-memory modetype config.pys dominationg (model would struggle to capture long-term dependencies)
# High R_MIN results in long-memory modes dominating (model may struggle to capture fast dynamics).
R_MIN = 0.9
R_MAX = 0.999