#-------------------------------------------------------
#            Code used to generate Fig. 7, 9, 10
#-------------------------------------------------------


import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'common')

import torch
from torch.autograd        import grad
from pykeops.torch         import Kernel

from divergences import kernel_divergence
from divergences import regularized_ot_visualization, hausdorff_divergence_visualization, sinkhorn_divergence_visualization

routines = {
    "kernel" : kernel_divergence,
    "regularized_ot_visualization" : regularized_ot_visualization,
    "hausdorff_visualization" : hausdorff_divergence_visualization,
    "sinkhorn_visualization" : sinkhorn_divergence_visualization,
}

import numpy as np
from scipy import misc
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from time  import time

use_cuda = torch.cuda.is_available()
tensor   = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor 

plt.ion()
plt.show()

s2v = lambda x : tensor([x])


# Load the pngs ================================================================================

dataset = "bars"
plot_springs = False

datasets = {
    "moons" : ("data/moon_a.png",  "data/moon_b.png"),
    "bars"  : ("data/bar_a.png",   "data/bar_b.png"),
}

from sampling import draw_samples, display_samples

# Alpha and Beta are probability measures sampled from two png densities
N, M = 500, 500 # Number of sample points for source and target

α_i, x_i = draw_samples(datasets[dataset][0], N, tensor)
β_j, y_j = draw_samples(datasets[dataset][1], M, tensor)
x_i.requires_grad = True

# Parameters of our data fidelity term  =========================================================
experiments = {}

if True : # Warmup PyTorch,  display dataset
    experiments["warmup"] = {
        "formula"     : "kernel",
        "k"           : ("energy", None), }
    experiments["dataset"] = {
        "formula"     : "kernel",
        "k"           : ("energy", None), }

if True : # Hausdorff
    for p in [1,2] : # C(x,y) = |x-y|^1 or |x-y|^2
        for eps, eps_s in [ (.01, "S"), (.05, "M"), (.1, "L")] :
            for nits in [1, 2, 10] :
                experiments["hausdorff_L{}_{}_{}its".format(p, eps_s, nits)] = {
                    "formula"        : "hausdorff_visualization",
                    "p"              : p,
                    "eps"            : eps**p, # Remember : eps is homogeneous to C(x,y)
                    "nits"           : nits,
                }

if True : # Sinkhorn
    for p in [1,2] : # C(x,y) = |x-y|^1 or |x-y|^2
        for eps, eps_s in [ (.01, "S"), (.1, "L") ] :
            for nits in [1.5, 2, 4.5, 5, 9.5, 10, 29.5, 30, 99.5, 100] :
                experiments["sinkhorn_L{}_{}_{}its".format(p, eps_s, nits)] = {
                    "formula"        : "sinkhorn_visualization",
                    "p"              : p,
                    "eps"            : eps**p, # Remember : eps is homogeneous to C(x,y)
                    "nits"           : nits,
                }


# Loop on all the experiments ================================================================

# We'll save the output wrt. the number of iterations
display = True
#plt.figure(figsize=(10,10))

os.makedirs(os.path.dirname("output/sinkhorn/{}/".format(dataset)), exist_ok=True)

def test(name, params, verbose=True) :
    params["kernel_heatmap_range"] = (0,1,100)
    # Compute the cost and gradient (+ fancy heatmaps in the background) ========================
    t_0 = time()
    loss, hmaps = routines[params["formula"]](α_i, x_i, β_j, y_j, heatmaps=True, **params)
    loss.backward()
    t_1 = time()
    if verbose : print("{} : {:.2f}s, cost = {:.6f}".format( name, t_1-t_0, loss.item()) )

    # Display ==================================================================================
    if display :
        plt.clf()

        plt.scatter( [10], [10] ) # shameless hack to prevent the slight pyplot change of axis...

        display_samples(plt.gca(), y_j, (.55,.55,.95))
        display_samples(plt.gca(), x_i, (.95,.55,.55), None if True else x_i.grad)
        hmaps.plot(plt.gca(), springs=plot_springs)
        
        plt.axis("equal")
        plt.axis([0,1,0.125,.875])
        plt.yticks([.2,.4,.6,.8])
        # Save result in the "output/" folder :
        plt.savefig("output/sinkhorn/" + dataset + "/" + name +".png") 
        plt.pause(.01)

    x_i.grad.zero_()

for name, params in experiments.items() :
    test(name, params)

print("Done.")
plt.show(block=True)
