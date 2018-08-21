#--------------------------------------------------------------------------------------------
#    Key routines of this repository, where we implement the Sinkhorn algorithms and MMDs
#--------------------------------------------------------------------------------------------

import numpy as np
import torch

# Reference implementations
from kernel_norm import kernel_divergence
from sinkhorn_balanced import regularized_ot, hausdorff_divergence, sinkhorn_divergence

# Simpler implementation, with less features - read it first
from sinkhorn_balanced_simple   import regularized_ot as regularized_ot_simple
from sinkhorn_balanced_simple   import hausdorff_divergence as hausdorff_divergence_simple
from sinkhorn_balanced_simple   import sinkhorn_divergence  as sinkhorn_divergence_simple

# Extended implementation, with extra features which we only use to generate the "Transport Plan" figures
from sinkhorn_balanced_visualization   import regularized_ot as regularized_ot_visualization
from sinkhorn_balanced_visualization   import hausdorff_divergence as hausdorff_divergence_visualization
from sinkhorn_balanced_visualization   import sinkhorn_divergence  as sinkhorn_divergence_visualization




