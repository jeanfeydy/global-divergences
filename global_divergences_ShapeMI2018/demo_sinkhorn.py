import torch
from torch.autograd        import grad
from sparse_distance_bmp   import sparse_distance_bmp
from pykeops.torch         import Kernel

import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
from time  import time

use_cuda = torch.cuda.is_available()
tensor   = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor 

plt.ion()
plt.show()

s2v = lambda x : tensor([x])


from scipy.ndimage.filters import gaussian_filter

def LoadImage(fname) :
    img = misc.imread(fname, flatten = True) # Grayscale
    img = gaussian_filter(img, 1, mode='nearest')
    img = (img[::-1, :])  / 255.
    img = np.swapaxes(img, 0,1 )
    return tensor( 1 - img )

# Load the pngs
dataset = "worms_c"
datasets = {
    "amoeba"   : ("data/amoeba_1.png",   "data/amoeba_2.png"),
    "dirac"    : ("data/dirac_a.png",    "data/dirac_b.png"),
    "blobs"    : ("data/blobs_a.png",    "data/blobs_b.png"),
    "knees"    : ("data/knee_1.png",     "data/knee_2.png"),
    "knees_2D" : ("data/knee_source.png","data/knee_target.png"),
    "bars"     : ("data/hor_bars_1.png", "data/hor_bars_2.png"),
    "bars_b"   : ("data/hor_bars_1.png", "data/hor_bars_1b.png"),
    "minibars" : ("data/minibar_1.png",  "data/minibar_2.png"),
    "keys"     : ("data/sol.png",        "data/fa.png"),
    "keys_n"   : ("data/sol_noisy.png",  "data/fa_noisy.png"),
    "worms"    : ("data/worm_1.png",     "data/worm_2.png"),
    "worms_a"  : ("data/worm_a.png",     "data/worm_b.png"),
    "worms_b"  : ("data/worm_1.png",     "data/worm_1b.png"),
    "worms_c"  : ("data/worm_c.png",     "data/worm_d.png"),
}
if dataset == "knees_2D" :
    sample = 8
    smooth = True # As the structure is very thin, sampling on a grid every 8 pixels looks weird...
                  # we smooth the gradient a little bit to prevent this visualization artifact from being visible
    scale_grad = 0.04
else :
    sample = 4
    smooth = False
    scale_grad = 0.02

source = LoadImage(datasets[dataset][0])
target = LoadImage(datasets[dataset][1])

# The images are rescaled to fit into the unit square ==========================================
scale = source.shape[0]
affine = tensor( [[ 1, 0, 0 ],
                  [ 0, 1, 0 ] ]) / scale

# Parameters of our data attachment term =======================================================
experiments = {}


if True : # Warmup PyTorch
    experiments["warmup"] = {
        "formula"     : "kernel",
        "id"          : Kernel( "-distance(x,y)" ),
        "gamma"       : s2v( 1. ),
    }


if True : # sinkhorn
    for nits in [1, 4, 9, 29, 100] :
        experiments["sinkhorn_L2_it_{}_plus_05".format(nits)] = {
            "formula"     : "sinkhorn",
            "epsilon"     : s2v( .01 ),
            "kernel"      : {
                "id"     : Kernel("gaussian(x,y)") ,
                "gamma"  :  1 / s2v( .01 )      },
            "tol"         : 1e-6,
            "nits"        : nits,
            "rho"         : -1., # Balanced transport. Make sure that both measures have equal mass!
            "transport_plan" : "minimal_symmetric+heatmaps",
            "end_on_target" : True,
        }
        experiments["sinkhorn_L2_it_{}_plus_1".format(nits)] = {
            "formula"     : "sinkhorn",
            "epsilon"     : s2v( .01 ),
            "kernel"      : {
                "id"     : Kernel("gaussian(x,y)") ,
                "gamma"  :  1 / s2v( .01 )      },
            "tol"         : 1e-6,
            "nits"        : nits+1,
            "rho"         : -1., # Balanced transport. Make sure that both measures have equal mass!
            "transport_plan" : "minimal_symmetric+heatmaps",
            "end_on_target" : False,
        }

if True : # sinkhorn_sym
    for nits in [1, 2, 3, 5, 10] :
        experiments["sinkhorn_sym_L2_it_{}".format(nits)] = {
            "formula"        : "hausdorff",
            "nits"           : nits,
            "id"          : Kernel( "gaussian(x,y)" ),
            "epsilon"     : s2v(   .01   ),
            "gamma"       : s2v( 1/.01),
            "only_a"      : True,
        }

if True : # hausdorff_sinkhorn
    for nits in [1, 2, 3, 5, 10] :
        experiments["hausdorff_L2_it_{}".format(nits)] = {
            "formula"        : "hausdorff",
            "nits"           : nits,
            "id"          : Kernel( "gaussian(x,y)" ),
            "epsilon"     : s2v(   .01   ),
            "gamma"       : s2v( 1/.01),
            "gradient"    : True,
        }

if True : # sinkhorn
    for nits in [ 100] :
        experiments["sinkhorn_L2_it_{}_plus_05".format(nits)] = {
            "formula"     : "sinkhorn",
            "epsilon"     : s2v( .01 ),
            "kernel"      : {
                "id"     : Kernel("gaussian(x,y)") ,
                "gamma"  :  1 / s2v( .01 )      },
            "tol"         : 1e-6,
            "nits"        : nits,
            "rho"         : -1., # Balanced transport. Make sure that both measures have equal mass!
            "transport_plan" : "minimal_symmetric+heatmaps",
            "gradient"    : True,
        }

# We'll save the output wrt. the number of iterations
display = True
plt.figure(figsize=(10,10))


def test(name, params, verbose=True) :
    params["kernel_heatmap_range"] = (0,1,100)

    # Compute the cost and gradient ============================================================
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

        # Source + Target :
        source_plot = .3*source.cpu().numpy()
        target_plot = .3*target.cpu().numpy()
        if not params.get("only_a", False ) :
            img_plot    = np.dstack( ( np.ones(source_plot.shape) - target_plot, 
                                    np.ones(source_plot.shape) - source_plot - target_plot, 
                                    np.ones(source_plot.shape) - source_plot ) )
        else :
            img_plot    = np.dstack( ( np.ones(source_plot.shape), 
                                    np.ones(source_plot.shape) - source_plot, 
                                    np.ones(source_plot.shape) - source_plot ) )
            if heatmaps is not None :
                heatmaps.b = None
        
        plt.imshow( np.swapaxes(img_plot,0,1), origin="lower", extent=(0,1,0,1))

        if params.get("gradient", False) :
            # Subsample the gradient field :
            grad_plot = grad_src.cpu().numpy()
            grad_plot = grad_plot[::sample, ::sample, :]
            if smooth :
                grad_plot = gaussian_filter(grad_plot, [1,1,0], mode='nearest')

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

        # "Distance" fields :
        if heatmaps is not None :
            heatmaps.plot(plt.gca(), {})
            
        # Save result in the "output/" folder :
        plt.savefig("output/sinkhorn/" + dataset + "_" + name +".png") 
        plt.pause(.01)


#test("hausdorff_KL_1_1", experiments["hausdorff_KL_1_1"], verbose=False) # one run for nothing
for name, params in experiments.items() :
    test(name, params)

print("Done.")
plt.show(block=True)
