"""Keep all configuration parameters in one place for easy access and modification."""

N_ROIS = 24
T = 240        # Number of timepoints. This is a dataset-level cocanstant and not a model param. T, _ = x.shape is used for the model
H = 4          # Per-node feature dimension (hidden multiplier to "lift" to higher dimension for Koopman linearity) Note: keep H << N_ROIS to avoid overfitting. Original paper used H=2
M = N_ROIS * H #T= = 240 Koopman latent dimension 

# Koopman eigenvalue initialization -> R_MIN and R_MAX control the minimum and maximum magnitude of eigenvalues at initialization
# Values close to 1.0 give slow dynamics, which is appropriate for fMRI data. Setting R_MAX < 1.0 ensures stability of the Koopman operator.
# Low R_MIN results in short-memory modetype config.pys dominationg (model would struggle to capture long-term dependencies)
# High R_MIN results in long-memory modes dominating (model may struggle to capture fast dynamics).
# See 3.3 of Resurrecting Recurrent Neural Networks for Long Sequences for hyperparameter tuning
R_MIN = 0.9
R_MAX = 0.999


# Hyperparams for Control Module encoder (row-wise MLP)
MLP_HIDDEN = 64 #MLP hidden multiplier
NHEAD = 4
NUM_LAYERS = 2

#Brick implementation
BETA = 0.5        # loss balance: β * L_cls + (1-β) * L_ELBO
NUM_CLASSES = 2   # number of task states for classifier (pre- vs post- sonication)
EPSILON = 0.1     # prior variance for g_0 ~ N(0, εI)