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
experiments["energy"] = {
    "formula" : "kernel",
    "k"    : ("energy", None) }
"""
for eps, eps_s in [ (.01, "S"), (.05, "M"), (.1, "L"), (.5, "XL"), (100., "XXL")] :
    experiments["gaussian_{}".format(eps_s)] = {
        "formula" : "kernel",
        "k"    : ("gaussian", eps) }
    experiments["laplacian_{}".format(eps_s)] = {
        "formula" : "kernel",
        "k"    : ("laplacian", eps) }
"""
for p in [2,1] :
    for eps, eps_s in [ (.01, "S"), (.05, "M"), (.1, "L"), (.5, "XL") ] :
        
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

def display(x, color, list_save=None) :
    kde  = KernelDensity(kernel='gaussian', bandwidth= .005 ).fit(x.data.cpu().numpy())
    dens = np.exp( kde.score_samples(t_plot) )
    dens[0] = 0 ; dens[-1] = 0;
    plt.fill(t_plot, dens, color=color)
    if list_save is not None :
        list_save.append(dens.ravel()) # We'll save a csv at the end
    
for name, params in experiments.items() :
    print("Experiment :", name)
    x_ts = [t_plot.ravel()]

    # Dataset =====================================================================

    # Alpha and Beta are uniform probability measures supported by intervals in [0,1]
    N, M = 5000, 5000 # Number of sample points for source and target

    t_i = torch.linspace(0, 1, N).type(dtype).view(-1,1) ; t_j = torch.linspace(0, 1, M).type(dtype).view(-1,1)
    x_i = 0.2 * t_i                                      ; y_j = 0.4 * t_j + 0.6
    α_i = torch.ones(N,1).type(dtype) / N                ; β_j = torch.ones(M,1).type(dtype) / M
    # We're going to perform gradient descent on Cost(Alpha, Beta) 
    # wrt. the positions x_i of the diracs masses that make up Alpha:
    x_i.requires_grad_(True)  

    os.makedirs(os.path.dirname("output/flow_1D/{}/".format(name)), exist_ok=True)
    for i in tqdm(range(Nsteps)): # Gradient flow ================================
        if i in save_its :
            display(y_j, (.55,.55,.95))
            display(x_i, (.95,.55,.55) , x_ts )
            plt.axis([-.1,1.1,-.1,5.5])

            plt.savefig("output/flow_1D/{}/{:03d}.png".format(name, i))
            plt.clf()
        
        # Compute cost
        loss = routines[params["formula"]](α_i, x_i, β_j, y_j, **params)

        # Compte gradient and update x_i
        loss.backward()
        x_i.data -= lr * (x_i.grad / α_i.data) # in-place modification of the tensor's values
        x_i.grad.zero_()

    # Save progress in a csv file for Tikz display in the paper ============================

    header = ""
    data = np.stack(x_ts).T
    np.savetxt("output/flow_1D/{}.csv".format(name), data, fmt='%-9.5f', header=header, comments = "")

print("Done.")