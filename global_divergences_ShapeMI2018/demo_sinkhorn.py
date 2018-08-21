#-------------------------------------------------------
#            Code used to generate Fig. 7, 9, 10
#-------------------------------------------------------


import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'common')

import torch
from torch.autograd        import grad
from sparse_distance_bmp   import sparse_distance_bmp
from pykeops.torch         import Kernel

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

def LoadImage(fname) :
    img = misc.imread(fname, flatten = True) # Grayscale
    #img = gaussian_filter(img, 1, mode='nearest')
    img = (img[::-1, :])  / 255.
    img = np.swapaxes(img, 0,1 )
    return tensor( 1 - img )

dataset = "bars"
datasets = {
    "knees_2D" : ("data/knee_a.png",  "data/knee_b.png"),  # Figs 1.c, 1.d, 
    "worms"    : ("data/worm_a.png",  "data/worm_b.png"),  # Figs 3, 4.c
    "blobs"    : ("data/blobs_a.png", "data/blobs_b.png"), # Figs 4.a, 4.b, 6
    "bars"     : ("data/bar_a.png",   "data/bar_b.png"),
}
if dataset == "knees_2D" :
    sample, scale_grad = 8, .03
    smooth = True # As the structure is very thin, sampling on a grid every 8 pixels looks weird...
                  # we smooth the gradient a little bit to prevent this visualization artifact from being visible
else :
    sample, scale_grad = 4, .015
    smooth = False

# Note that both measures will be normalized in "sparse_distance_bmp"
source = LoadImage(datasets[dataset][0])
target = LoadImage(datasets[dataset][1])

# The images are rescaled to fit into the unit square ==========================================
scale = source.shape[0]
affine = tensor( [[ 1, 0, 0 ],
                  [ 0, 1, 0 ] ]) / scale

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
    for p in [1, 2] : # C(x,y) = |x-y|^1 or |x-y|^2
        for eps, eps_s in [ (.01, "S"), (.05, "M"), (.1, "L"), (.5, "XL"), (100., "XXL")] :
            for nits in [1, 2, 5, 10, 30] :
                experiments["hausdorff_L{}_{}_{}its".format(p, eps_s, nits)] = {
                    "formula"        : "hausdorff_visualization",
                    "p"              : p,
                    "eps"            : eps**p, # Remember : eps is homogeneous to C(x,y)
                    "nits"           : nits,
                }

if True : # Sinkhorn
    for p in [1, 2] : # C(x,y) = |x-y|^1 or |x-y|^2
        for eps, eps_s in [ (.01, "S"), (.1, "L"), (100., "XXL") ] :
            for nits in [1.5, 2, 4.5, 5, 9.5, 10, 29.5, 30] :
                experiments["sinkhorn_L{}_{}_{}its".format(p, eps_s, nits)] = {
                    "formula"        : "sinkhorn_visualization",
                    "p"              : p,
                    "eps"            : eps**p, # Remember : eps is homogeneous to C(x,y)
                    "nits"           : nits,
                }


# Loop on all the experiments ================================================================

# We'll save the output wrt. the number of iterations
display = True
plt.figure(figsize=(10,10))


def test(name, params, verbose=True) :
    params["kernel_heatmap_range"] = (0,1,100)
    # Compute the cost and gradient (+ fancy heatmaps in the background) ========================
    t_0 = time()
    cost, grad_src, heatmaps = sparse_distance_bmp(params, source, target, 
                                                           affine, affine, 
                                                           normalize=True, info=display )
    t_1 = time()
    if verbose : print("{} : {:.2f}s, cost = {:.6f}".format( name, t_1-t_0, cost.item()) )

    # Display ==================================================================================
    grad_src = - grad_src # N.B.: We want to visualize a descent direction, not the opposite!
    if display :
        plt.clf()

        # Source + Target : fancy  red+blue overlay
        source_plot = .3*source.cpu().numpy()
        target_plot = .3*target.cpu().numpy()
        img_plot    = np.dstack( ( np.ones(source_plot.shape) - target_plot, 
                                   np.ones(source_plot.shape) - source_plot - target_plot, 
                                   np.ones(source_plot.shape) - source_plot ) )
        
        plt.imshow( np.swapaxes(img_plot,0,1), origin="lower", extent=(0,1,0,1))

        if name != "dataset" :
            # Subsample the gradient field :
            grad_plot = grad_src.cpu().numpy()
            grad_plot = grad_plot[::sample, ::sample, :]
            if smooth :
                grad_plot = gaussian_filter(grad_plot, [2,2,0], mode='nearest')

            # Display the gradient field :
            X,Y   = np.meshgrid( np.linspace(0, 1, grad_plot.shape[0]+1)[:-1] + .5/(sample*grad_plot.shape[0]), 
                                np.linspace(0, 1, grad_plot.shape[1]+1)[:-1] + .5/(sample*grad_plot.shape[1]) )
            U, V  = grad_plot[:,:,0], grad_plot[:,:,1]
            norms = np.sqrt( (grad_plot[:,:,1]**2 + grad_plot[:,:,0]**2) )
            mask  = (norms>0)
            scale = np.mean( norms[mask] )
            if smooth : mask  = (source_plot[::sample, ::sample]>0)

            Y, X, U, V = Y[mask], X[mask], U[mask], V[mask]
            plt.quiver( Y, X, U, V, 
                        scale = scale_grad*scale, scale_units="dots", color="#5CBF3A", zorder=3, width=0.0025)

            # "Distance", influence fields in the background :
            if heatmaps is not None :
                heatmaps.plot(plt.gca())
            
        # Save result in the "output/" folder :
        plt.savefig("output/sinkhorn/" + dataset + "_" + name +".png") 
        plt.pause(.01)


for name, params in experiments.items() :
    test(name, params)

print("Done.")
plt.show(block=True)
