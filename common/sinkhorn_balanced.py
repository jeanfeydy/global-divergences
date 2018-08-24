#--------------------------------------------------------------------------------------------
#    Key routines of this repository, where we implement the Sinkhorn algorithms on probability measures
#
# On top of the barebone algorithms, this file provides :
# - Two backends : KeOps (efficient) and vanilla PyTorch (simple)
# - Bypassing of the autograd mechanism, if we assume convergence in the Sinkhorn loop (2-3x speed). 
# - Fancy visualizations in the background, through the Heatmaps class.
# 
# This code is therefore much longer than a straightforward CopyPaste of our paper.
# A compact implementation of Sinkhorn divergences should fit in about ~40 lines
# of PyTorch + Keops code: feel free to copy this reference implementation, 
# and remove the bits that you do not need.
#
#--------------------------------------------------------------------------------------------

import numpy as np
import torch
from display import Heatmaps, grid

try :
    from pykeops.torch import generic_sum, generic_logsumexp
    backend = "keops"   # Efficient GPU backend, which scales up to ~1,000,000 samples.
except ImportError :
    backend = "pytorch" # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!


#######################################################################################################################
# Elementary operations .....................................................................
#######################################################################################################################

def scal( α, f ) :
    return torch.dot( α.view(-1), f.view(-1) )

def lse( v_ij ):
    """[lse(v_ij)]_i = log sum_j exp(v_ij), with numerical accuracy."""
    V_i = torch.max(v_ij, 1)[0].view(-1,1)
    return V_i + (v_ij - V_i).exp().sum(1).log().view(-1,1)

def Sinkhorn_ops(p, ε, x_i, y_j) :
    """
    Given:
    - an exponent p = 1 or 2
    - a regularization strength ε > 0
    - point clouds x_i and y_j, encoded as N-by-D and M-by-D torch arrays,

    Returns a pair of routines S_x, S_y such that
      [S_x(f_i)]_j = -log sum_i exp( f_i - |x_i-y_j|^p / ε )
      [S_y(f_j)]_i = -log sum_j exp( f_j - |x_i-y_j|^p / ε )

    This may look like a strange level of abstraction, but it is the most convenient way of
    working with KeOps and Vanilla pytorch (with a pre-computed cost matrix) at the same time.
    """
    if   backend == "keops" : # Memory-efficient GPU implementation : ONline logsumexp
        # We create a KeOps GPU routine...
        if   p == 1 : formula = "Fj - (Sqrt(SqDist(Xi,Yj))  / E)"
        elif p == 2 : formula = "Fj -      (SqDist(Xi,Yj)   / E)"
        else :        
            formula = "Fj - (Powf(SqDist(Xi,Yj),R)/ E)"
            raise(NotImplementedError("I should fix the derivative at 0 of Powf, in KeOps's core."))
        D = x_i.shape[1] # Dimension of the ambient space (typically 2 or 3)
        routine = generic_logsumexp( formula, "outi = Vx(1)", # Formula, output...
            # and input variables : ε, x_i, y_j, f_j, p/2 given with their respective dimensions
            "E = Pm(1)", "Xi = Vx({})".format(D), "Yj = Vy({})".format(D), "Fj = Vy(1)", "R=Pm(1)")

        # Before wrapping it up in a simple pair of operators - don't forget the minus!
        ε,r = torch.Tensor([ε]).type_as(x_i), torch.Tensor([p/2]).type_as(x_i)
        S_x = lambda f_i : -routine(ε, y_j, x_i, f_i, r)
        S_y = lambda f_j : -routine(ε, x_i, y_j, f_j, r)
        return S_x, S_y

    elif backend == "pytorch" : # Naive matrix-vector implementation : OFFline logsumexp
        # We precompute the |x_i-y_j|^p matrix once and for all...
        C_e  = ( (x_i.unsqueeze(1) - y_j.unsqueeze(0)) ** 2).sum(2)
        if   p == 1 : C_e = C_e.sqrt() / ε
        elif p == 2 : C_e = C_e / ε
        else : C_e = C_e**(p/2) / ε
        CT_e = C_e.t()

        # Before wrapping it up in a simple pair of operators - don't forget the minus!
        S_x = lambda f_i : -lse( f_i.view(1,-1) - CT_e )
        S_y = lambda f_j : -lse( f_j.view(1,-1) - C_e  )
        return S_x, S_y


#######################################################################################################################
# Sinkhorn iterations .....................................................................
#######################################################################################################################

def sink(α_i, x_i, β_j, y_j, p=1, eps=.1, nits=100, tol=1e-3, assume_convergence=False, heatmaps=None, **kwargs):

    ε = eps # Python supports Unicode. So fancy!
    if type(nits) in [list, tuple] : nits = nits[0]  # The user may give different limits for Sink and SymSink
    # Sinkhorn loop with A = a/eps , B = b/eps ....................................................
    
    α_i_log, β_j_log = α_i.log(), β_j.log() # Precompute the logs of the measures' weights
    B_i,     A_j     = torch.zeros_like(α_i), torch.zeros_like(β_j) # Sampled influence fields
    
    # if we assume convergence, we can skip all the "save computational history" stuff
    torch.set_grad_enabled(not assume_convergence)

    S_x, S_y = Sinkhorn_ops(p, ε, x_i, y_j) # Softmin operators (divided by ε, as it's slightly cheaper...)
    for i in range(nits-1):
        B_i_prev = B_i

        A_j = S_x( B_i + α_i_log )   # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε
        B_i = S_y( A_j + β_j_log )   # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε

        err = ε * (B_i - B_i_prev).abs().mean() # Stopping criterion: L1 norm of the updates
        if err.item() < tol : break
    

    # Fancy display, in the background... 
    if heatmaps == True :
        global grid
        grid = grid.type_as(x_i)
        S2_x, _ = Sinkhorn_ops(p, ε, x_i, grid)
        _, S2_y = Sinkhorn_ops(p, ε, grid, y_j)
        # Note that we want the final B_i and B_grid to coincide on x_i,
        # so we must emulate with "A2_j" the alternate updates of the Sinkhorn loop
        A_grid = S2_x( B_i + α_i_log )  # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε
        A2_j   =  S_x( B_i + α_i_log )
        B_grid = S2_y( A2_j + β_j_log ) # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε

        hmaps = (ε*A_grid.view(-1), ε*B_grid.view(-1))
    else : hmaps = None, None


    torch.set_grad_enabled(True)
    # One last step, which allows us to bypass PyTorch's backprop engine if required (as explained in the paper)
    if not assume_convergence :
        A_j = S_x( B_i + α_i_log )
        B_i = S_y( A_j + β_j_log )
    else : # Assume that we have converged, and can thus use the "exact" (and cheap!) gradient's formula
        S_x, _ = Sinkhorn_ops(p, ε, x_i.detach(), y_j)
        _, S_y = Sinkhorn_ops(p, ε, x_i, y_j.detach())
        A_j = S_x( (B_i + α_i_log).detach() )
        B_i = S_y( (A_j + β_j_log).detach() )

    a_y, b_x = ε*A_j.view(-1), ε*B_i.view(-1)
    return a_y, b_x, hmaps


def sym_sink(α_i, x_i, y_j=None, p=1, eps=.1, nits=100, tol=1e-3, assume_convergence=False, heatmaps=None, **kwargs):

    ε = eps # Python supports Unicode. So fancy!
    if type(nits) in [list, tuple] : nits = nits[1]  # The user may give different limits for Sink and SymSink
    # Sinkhorn loop ......................................................................

    α_i_log = α_i.log()
    A_i     = torch.zeros_like(α_i)
    S_x, _  = Sinkhorn_ops(p, ε, x_i, x_i) # Sinkhorn operator from x_i to x_i (divided by ε, as it's slightly cheaper...)
    
    # if we assume convergence, we can skip all the "save computational history" stuff
    torch.set_grad_enabled(not assume_convergence)

    for i in range(nits-1):
        A_i_prev = A_i

        A_i = 0.5 * (A_i + S_x(A_i + α_i_log) ) # a(x)/ε = .5*(a(x)/ε + Smin_ε,y~α [ C(x,y) - a(y) ] / ε)
        
        err = ε * (A_i - A_i_prev).abs().mean()    # Stopping criterion: L1 norm of the updates
        if err.item() < tol : break
    
    # Fancy display, in the background... 
    if heatmaps == True :
        global grid
        grid = grid.type_as(x_i)
        S2_x, _ = Sinkhorn_ops(p, ε, x_i, grid)
        A_grid = S2_x( A_i + α_i_log ) # a(y)/ε = Smin_ε,x~α [ C(x,y) - a(x) ]  / ε
        a_grid = ε*A_grid.view(-1)
    else : a_grid = None


    torch.set_grad_enabled(True)
    # One last step, which allows us to bypass PyTorch's backprop engine if required
    if not assume_convergence :
        W_i = A_i + α_i_log
        S2_x, _ = Sinkhorn_ops(p, ε, x_i, y_j) # Sinkhorn operator from x_i to y_j (divided by ε...)
    else :
        W_i = (A_i + α_i_log).detach()
        S_x,  _ = Sinkhorn_ops(p, ε, x_i.detach(), x_i)
        S2_x, _ = Sinkhorn_ops(p, ε, x_i.detach(), y_j)

    a_x = ε * S_x( W_i ).view(-1) # a(x) = Smin_e,z~α [ C(x,z) - a(z) ]
    if y_j is None : 
        return None, a_x, a_grid
    else : # extrapolate "a" to the point cloud "y_j"
        a_y = ε * S2_x( W_i ).view(-1) # a(z) = Smin_e,z~α [ C(y,z) - a(z) ]
        return a_y, a_x, a_grid


#######################################################################################################################
# Derived Functionals .....................................................................
#######################################################################################################################

def  regularized_ot( α, x, β, y, **params): # OT_ε
    a_y, b_x, hmaps = sink( α, x, β, y, **params)
    cost = scal(α, b_x) + scal(β, a_y)

    return cost if (params.get("heatmaps", None) is None) else (cost, Heatmaps(*hmaps)) 
        

def hausdorff_divergence(α, x, β, y, **params): # H_ε
    a_y, a_x, a_grid = sym_sink( α, x, y,    **params)
    b_x, b_y, b_grid = sym_sink( β, y, x,    **params)
    cost = .5 * ( scal( α, b_x - a_x ) + scal( β, a_y - b_y ) )

    return cost if (params.get("heatmaps", None) is None) else (cost, Heatmaps(a_grid,b_grid)) 


def  sinkhorn_divergence(α, x, β, y, **params): # S_ε
    a_y, b_x, hmaps =     sink( α, x, β, y, **params)
    _,   a_x, _     = sym_sink( α, x,       **params )
    _,   b_y, _     = sym_sink( β, y,       **params )
    cost =  scal( α, b_x - a_x ) + scal( β, a_y - b_y )

    return cost if (params.get("heatmaps", None) is None) else (cost, Heatmaps(*hmaps)) 





