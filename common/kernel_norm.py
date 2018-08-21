#--------------------------------------------------------------------------------------------
#    Implements an MMD, or kernel squared norm
#--------------------------------------------------------------------------------------------

import numpy as np
import torch
from display import Heatmaps, grid

try :
    from pykeops.torch import generic_sum, generic_logsumexp
    backend = "keops"   # Efficient GPU backend, which scales up to ~1,000,000 samples.
except ImportError :
    backend = "pytorch" # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!

#########################################################################################################################
# MMD Loss --------------------------------------------------------------------------------------------------------------
#########################################################################################################################

def scal( α, f ) :
    return torch.dot( α.view(-1), f.view(-1) )

def conv(k, x_i, y_j, β_j) :
    k_name, s = k
    if   backend == "keops" : # Memory-efficient GPU implementation : ONline map-reduce
        # We create a KeOps GPU routine...
        s2v = lambda g : torch.Tensor([g]).type_as(x_i)
        if   k_name == "energy"   : formula = " - Sqrt(SqDist(Xi,Yj))  * Bj" ; g = s2v(1.)
        elif k_name == "gaussian" : formula = "Exp( -G*SqDist(Xi,Yj) ) * Bj" ; g = s2v(1/s**2)
        elif k_name == "laplacian": formula = "Exp( -G*Sqrt(SqDist(Xi,Yj)) ) * Bj" ; g = s2v(1/s)
        else :                      raise NotImplementedError()

        D = x_i.shape[1] # Dimension of the ambient space (typically 2 or 3)
        routine = generic_sum( formula, "out_i = Vx(1)", # Formula, output...
            # and input variables : g, x_i, y_j, β_j, given with their respective dimensions
            "G = Pm(1)", "Xi = Vx({})".format(D), "Yj = Vy({})".format(D), "Bj = Vy(1)")

        # ...Before applying it to our data:
        return routine(g, x_i, y_j, β_j)

    elif backend == "pytorch" : # Naive matrix-vector implementation : OFFline map-reduce
        XmY2 = ( (x_i.unsqueeze(1) - y_j.unsqueeze(0)) ** 2).sum(2)
        if   k_name == "energy"   : K =  -XmY2.sqrt()
        elif k_name == "gaussian" : K = (-XmY2 / s**2).exp()
        elif k_name == "laplacian": K = (-XmY2.sqrt() / s).exp()
        else :                      raise NotImplementedError()
        return K @ β_j

def kernel_divergence(α, x, β, y, k=("energy", None), heatmaps=None, **params):
    ma_x = conv(k, x, x, α) # -a = k ★ α
    mb_x = conv(k, x, y, β) # -b = k ★ β
    mb_y = conv(k, y, y, β) # -b = k ★ β
    # We now take advantage of the fact that ⟨α, k★β⟩ = ⟨β, k★α⟩
    # N.B.: I should also implement an "autograd trick",
    #       as we know that k is symmetric.
    cost = .5 * ( scal( α, ma_x - 2*mb_x ) + scal( β, mb_y ) )
    
    if heatmaps is None :
        return cost
    elif heatmaps == True : # display  k ★ (α-β) in the background
        global grid
        grid = grid.type_as(x)
        heats = Heatmaps( conv(k, grid, x, α) - conv(k, grid, y, β) )
        return cost, heats
    else :
        return cost, None

