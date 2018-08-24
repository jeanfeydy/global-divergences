#-------------------------------------------------------
#            Code used to generate Fig. ...
#-------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import torch
from tqdm import tqdm # Fancy progress bar

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'common')

from divergences import kernel_divergence, regularized_ot, hausdorff_divergence, sinkhorn_divergence
routines = {
    "kernel" : kernel_divergence,
    "regularized_ot" : regularized_ot,
    "hausdorff" : hausdorff_divergence,
    "sinkhorn" : sinkhorn_divergence,
}

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Parameters for the experiments ===============================================
experiments = {}
experiments["dataset"] = {
    "formula" : "kernel",
    "k"    : ("energy", None) }
experiments["energy"] = {
    "formula" : "kernel",
    "k"    : ("energy", None) }

for eps, eps_s in [ (.01, "S"), (.05, "M"), (.1, "L"), (.5, "XL"), (100., "XXL")] :
    experiments["gaussian_{}".format(eps_s)] = {
        "formula" : "kernel",
        "k"    : ("gaussian", eps) }
    experiments["laplacian_{}".format(eps_s)] = {
        "formula" : "kernel",
        "k"    : ("laplacian", eps) }

for p in [1,2] :
    for eps, eps_s in [ (.01, "S"), (.05, "M"), (.1, "L"), (.5, "XL"), (1., "XXL") ] :
        
        experiments["regularized_ot_L{}_{}".format(p, eps_s)] = {
            "formula" : "regularized_ot",
            "p"    : p,      # C(x,y) = |x-y|^p
            "eps"  : eps**p, # Regularization strength, homogeneous to |x-y|^p
            "tol"  : 1e-5,   # Tolerance - min L1 norm of the updates to break the loop
            "assume_convergence" : True,
        }
        experiments["hausdorff_L{}_{}".format(p, eps_s)] = {
            "formula" : "hausdorff",
            "p"    : p,      # C(x,y) = |x-y|^p
            "eps"  : eps**p, # Regularization strength, homogeneous to |x-y|^p
            "tol"  : 1e-5,   # Tolerance - min L1 norm of the updates to break the loop
        }
        experiments["sinkhorn_L{}_{}".format(p, eps_s)] = {
            "formula" : "sinkhorn",
            "p"    : p,      # C(x,y) = |x-y|^p
            "eps"  : eps**p, # Regularization strength, homogeneous to |x-y|^p
            "tol"  : 1e-5,   # Tolerance - min L1 norm of the updates to break the loop
            "assume_convergence" : True,
        }

# Gradient flow + display =====================================================
Nsteps, lr  = 501, .01 # Parameters for the gradient descent
t_plot      = np.linspace(-0.1, 1.1, 1000)[:,np.newaxis]
save_its    = [0, 25, 50, 100, 500]


# Dataset =====================================================================

from sampling import draw_samples, display_samples

# Alpha and Beta are probability measures sampled from two png densities
N, M = 500, 500 # Number of sample points for source and target

dataset = "crescents"
datasets = {
    "moons"     : ("data/moon_a.png",    "data/moon_b.png"),
    "densities" : ("data/density_a.png", "data/density_b.png"),
    "slopes"    : ("data/slope_a.png",   "data/slope_b.png"),
    "crescents" : ("data/crescent_a.png","data/crescent_b.png"),
}
if dataset == "crescents" :
    ax_limits = [0,1,0,1]
    y_ticks   = [0.,.2,.4,.6,.8, 1.]
else :
    ax_limits = [0,1,0.125,.875]
    y_ticks   = [.2,.4,.6,.8]
display_grad = False # True if you want the green arrows


α_i, X_i = draw_samples(datasets[dataset][0], N, dtype)
β_j, Y_j = draw_samples(datasets[dataset][1], M, dtype)


    
for name, params in experiments.items() :
    print("Experiment :", name)

    x_i = X_i.clone() ; y_j = Y_j.clone()

    # We're going to perform gradient descent on Cost(Alpha, Beta) 
    # wrt. the positions x_i of the diracs masses that make up Alpha:
    x_i.requires_grad_(True)  

    os.makedirs(os.path.dirname("output/flow_2D/{}/{}/".format(dataset,name)), exist_ok=True)
    for i in tqdm(range(Nsteps)): # Gradient flow ================================        
        # Compute cost
        loss = routines[params["formula"]](α_i, x_i, β_j, y_j, **params)

        # Compte gradient and display
        loss.backward()
        if i in save_its :
            plt.scatter( [10], [10] ) # shameless hack to prevent the slight pyplot change of axis...

            display_samples(plt.gca(), y_j, (.55,.55,.95))
            display_samples(plt.gca(), x_i, (.95,.55,.55), None if (name == "dataset" or not display_grad) else x_i.grad)

            plt.axis("equal")
            plt.axis(ax_limits)
            plt.yticks(y_ticks)
            plt.gca().set_aspect('equal', adjustable='box')

            plt.savefig("output/flow_2D/{}/{}/{:03d}.png".format(dataset, name, i))
            plt.clf()
        
        if name != "dataset" :
            # update the x_i's
            x_i.data -= lr * (x_i.grad / α_i.data) # in-place modification of the tensor's values
            x_i.grad.zero_()


print("Done.")