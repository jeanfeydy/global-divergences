#-------------------------------------------------------
#            Code used to generate Fig. ...
#-------------------------------------------------------

import numpy as np
import torch

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

print("Use cuda : ", use_cuda)

# Parameters for the experiments ===============================================
experiments = {}
experiments["dataset"] = {
    "formula" : "kernel",
    "k"    : ("energy", None) }
experiments["energy"] = {
    "formula" : "kernel",
    "k"    : ("energy", None) }

epsilons = [.001, .002, .004, .007, .01, .02, .04, .07, .1, .2, .4, .7, 1.]
for p in [1] :
    for tol in [1e-3, 1e-5, 1e-7] :
        for eps in epsilons :
            eps = float(eps) # Weird pytorch bug...
            experiments["sinkhorn_L{}_{:05.3f}_tol_{:.2E}".format(p, eps, tol)] = {
                "formula" : "sinkhorn",
                "p"    : p,      # C(x,y) = |x-y|^p
                "eps"  : eps**p, # Regularization strength, homogeneous to |x-y|^p
                "nits" : 1000,
                "tol"  : tol,   # Tolerance - min L1 norm of the updates to break the loop
                "assume_convergence" : True,
            }

# Dataset =====================================================================
from random import choices
from scipy import misc

def LoadImage(fname) :
    img = misc.imread(fname, flatten = True) # Grayscale
    img = (img[::-1, :])  / 255.
    #img = np.swapaxes(img, 0,1 )
    return 1 - img

def DrawSamples(fname, n) :
    A = LoadImage(fname)
    xg, yg = np.meshgrid( np.linspace(0,1,A.shape[0]), np.linspace(0,1,A.shape[1]) )
    
    grid = list( zip(xg.ravel(), yg.ravel()) )
    dens = A.ravel() / A.sum()
    dots = np.array( choices(grid, dens, k=n ) )
    dots += .005 * np.random.standard_normal(dots.shape)

    weights = torch.ones(n,1).type(dtype) / n
    return weights, torch.from_numpy(dots).type(dtype)

# Alpha and Beta are probability measures sampled from two png densities
N, M = 10000, 10000 # Number of sample points for source and target

α_i, X_i = DrawSamples("data/density_a.png", N)
β_j, Y_j = DrawSamples("data/density_b.png", M)

# Benchmark =====================================================================

loops = 10
import time, timeit
import gc
GC = 'gc.enable();' if True else 'pass;'

elapsed_times = { 1e-3 : [], 1e-5 : [], 1e-7 : [] }

for name, params in experiments.items() :
    print("Experiment {} : {:3}x".format(name, loops), end= "")

    x_i = X_i.clone() ; y_j = Y_j.clone()
    x_i.requires_grad_(True)  

    code = '''
loss = routines[params["formula"]](α_i, x_i, β_j, y_j, **params)
loss.backward()
x_i.grad.zero_()
    '''
    exec(code, locals())
    elapsed = timeit.Timer(code, GC,  
                            globals = locals(), timer = time.time).timeit(loops)

    print("{:3.6f}s".format(elapsed/loops))
    if params.get("tol", None) in elapsed_times.keys() :
        elapsed_times[params["tol"]].append(elapsed/loops)

header = "epsilon em3 em5 em7"
lines = [ epsilons, elapsed_times[1e-3], elapsed_times[1e-5], elapsed_times[1e-7] ]
benchs = np.array(lines).T
np.savetxt("output/benchmarks/epsilon_N{}_M{}.csv".format(N,M), benchs, 
            fmt='%-9.5f', header=header, comments='')

print("Done.")


