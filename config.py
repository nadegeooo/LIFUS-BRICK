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
MLP_HIDDEN = 256 #MLP hidden multiplier
NHEAD = 4
NUM_LAYERS = 4

#Brick implementation
BETA = 0             # loss balance: β * L_cls + (1-β) * L_ELBO
NUM_CLASSES = 2         # number of task states for classifier (pre- vs post- sonication)
EPSILON = 1.0           # prior variance for g_0 ~ N(0, εI)
LAMBDA_NOISE = 0.01     # noise scaling. 0.01 is ideal

# KL Annealing
KL_G0_DELAY_EPOCHS  = 0   # hold KL_g0 at 0 for first epochs
KL_G0_ANNEAL_EPOCHS = 0  # ramp KL_g0 from 0 to 1 over time
KL_G0_FREE_BITS     = 0  # minimum KL_g0 per dim before penalty kicks in
KL_U_DELAY_EPOCHS   = 0   # hold KL_u at 0 for first epochs
KL_U_ANNEAL_EPOCHS  = 0   # then ramp KL_u from 0 to 1 over time
KL_U_FREE_BITS      = 0  # minimum KL_u before penalty kicks in
U_PRIOR_SIGMA       = 0.5  # prior std on u_t (tighter = harder to collapse)

#Training
PATIENCE            = 30          # Epochs to wait for early stopping
WEIGHT_DECAY        = 0.05        # Weight decay for optimizer
BATCH_SIZE          = 12          # Batch size for training

OVERFIT_THRESHOLD = 1.5  # val/train recon ratio above which overfitting is detected