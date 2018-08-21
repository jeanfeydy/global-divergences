#--------------------------------------------------------------------------------------------
# On top of the reference implementation, this file provides :
# - Extra-Fancy visualizations in the background, through the Heatmaps class.
#--------------------------------------------------------------------------------------------

import numpy as np
import torch
from display import HeatmapsSprings, grid

try :
    from pykeops.torch import generic_sum, generic_logsumexp
    backend = "keops"   # Efficient GPU backend, which scales up to ~1,000,000 samples.
except ImportError :
    backend = "pytorch" # Vanilla torch backend. Beware of memory overflows above ~10,000 samples!


#######################################################################################################################
# Elementary operations .....................................................................
#######################################################################################################################

from sinkhorn_balanced import scal, lse, Sinkhorn_ops

def Barycenters_ops(p, ε, x_i, y_j) :
    """
    Given:
    - an exponent p = 1 or 2
    - a regularization strength ε > 0
    - point clouds x_i and y_j, encoded as N-by-D and M-by-D torch arrays,

    Returns a pair of routines R_x, R_y such that
      [R_x(f_i, g_j)]_j = sum_i exp( f_i + g_j - |x_i-y_j|^p / ε ) * (x_i-y_j)
      [R_y(f_j, g_i)]_i = sum_j exp( f_j + g_i - |x_i-y_j|^p / ε ) * (y_j-x_i)

    This may look like a strange level of abstraction, but it is the most convenient way of
    working with KeOps and Vanilla pytorch (with a pre-computed cost matrix) at the same time.
    """
    if   backend == "keops" : # Memory-efficient GPU implementation
        # We create a KeOps GPU routine...
        if   p == 1 : formula = "Exp(Fj + Gi - (Sqrt(SqDist(Xi,Yj))/ E)) * (Yj-Xi)"
        elif p == 2 : formula = "Exp(Fj + Gi -      (SqDist(Xi,Yj) / E)) * (Yj-Xi)"
        else :        
            formula = "Exp( Fj + Gi - (Powf(SqDist(Xi,Yj),R)/ E) ) * (Yj-Xi)"
            raise(NotImplementedError("I should fix the derivative at 0 of Powf, in KeOps's core."))
        D = x_i.shape[1] # Dimension of the ambient space (typically 2 or 3)
        routine = generic_sum( formula, "outi = Vx({})".format(D), # Formula, output...
            # and input variables : ε, x_i, y_j, f_j, p/2 given with their respective dimensions
            "E = Pm(1)", "Xi = Vx({})".format(D), "Yj = Vy({})".format(D), 
            "Fj = Vy(1)", "Gi = Vx(1)", "R=Pm(1)")

        # Before wrapping it up in a simple pair of operators - don't forget the minus!
        ε,r = torch.Tensor([ε]).type_as(x_i), torch.Tensor([p/2]).type_as(x_i)
        R_x = lambda f_i, g_j : routine(ε, y_j, x_i, f_i, g_j, r)
        R_y = lambda f_j, g_i : routine(ε, x_i, y_j, f_j, g_i, r)
        return R_x, R_y

    elif backend == "pytorch" :
        raise NotImplementedError()

def Projection_ops(p, ε, x_i, y_j) :
    "Normalization weights for the Barycenter ops."
    if   backend == "keops" : # Memory-efficient GPU implementation
        # We create a KeOps GPU routine...
        if   p == 1 : formula = "Exp(Fj + Gi - (Sqrt(SqDist(Xi,Yj))/ E))"
        elif p == 2 : formula = "Exp(Fj + Gi -      (SqDist(Xi,Yj) / E))"
        else :        
            formula = "Exp( Fj + Gi - (Powf(SqDist(Xi,Yj),R)/ E) )"
            raise(NotImplementedError("I should fix the derivative at 0 of Powf, in KeOps's core."))
        D = x_i.shape[1] # Dimension of the ambient space (typically 2 or 3)
        routine = generic_sum( formula, "outi = Vx(1)", # Formula, output...
            # and input variables : ε, x_i, y_j, f_j, p/2 given with their respective dimensions
            "E = Pm(1)", "Xi = Vx({})".format(D), "Yj = Vy({})".format(D), 
            "Fj = Vy(1)", "Gi = Vx(1)", "R=Pm(1)")

        # Before wrapping it up in a simple pair of operators - don't forget the minus!
        ε,r = torch.Tensor([ε]).type_as(x_i), torch.Tensor([p/2]).type_as(x_i)
        P_x = lambda f_i, g_j : routine(ε, y_j, x_i, f_i, g_j, r)
        P_y = lambda f_j, g_i : routine(ε, x_i, y_j, f_j, g_i, r)
        return P_x, P_y

    elif backend == "pytorch" :
        raise NotImplementedError()



#######################################################################################################################
# Sinkhorn iterations .....................................................................
#######################################################################################################################

def sink(α_i, x_i, β_j, y_j, p=1, eps=.1, nits=100, heatmaps=None, **kwargs):

    ε = eps # Python supports Unicode. So fancy!
    # Sinkhorn loop with A = a/eps , B = b/eps ....................................................
    
    α_i_log, β_j_log = α_i.log(), β_j.log() # Precompute the logs of the measures' weights
    B_i,     A_j     = torch.zeros_like(α_i), torch.zeros_like(β_j) # Sampled influence fields
    
    S_x, S_y = Sinkhorn_ops(p, ε, x_i, y_j) # Softmin operators (divided by ε, as it's slightly cheaper...)
    for i in range( int(2*(nits-1)) ):

        if i%2==0 : A_j = S_x( B_i + α_i_log )   # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε
        else      : B_i = S_y( A_j + β_j_log )   # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε


    # Fancy display, in the background... 
    if heatmaps == True :
        # Heatmaps --------------------------------------------------------------
        global grid
        grid = grid.type_as(x_i)
        S2_x, _ = Sinkhorn_ops(p, ε, x_i, grid)
        _, S2_y = Sinkhorn_ops(p, ε, grid, y_j)

        if i%2 == 1 :
            # Note that we want the final B_i and B_grid to coincide on x_i,
            # so we must emulate with "A2_j" the alternate updates of the Sinkhorn loop
            A_grid = S2_x( B_i + α_i_log )  # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε
            A2_j   =  S_x( B_i + α_i_log )
            B_grid = S2_y( A2_j + β_j_log ) # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε
        else : 
            B_grid = S2_y( A_j + β_j_log ) # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε
            B2_i   =  S_y( A_j + β_j_log )
            A_grid = S2_x( B2_i + α_i_log )  # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε

    # Final step
    if i%2 == 1 :
        A_j = S_x( B_i + α_i_log )
        B_i = S_y( A_j + β_j_log )
    else :
        B_i = S_y( A_j + β_j_log )
        A_j = S_x( B_i + α_i_log )
        
    if heatmaps == True :
        Rx, Ry = Barycenters_ops(p, ε, x_i, y_j)

        xt_i = x_i + Ry( A_j + β_j_log, B_i )
        yt_j = y_j + Rx( B_i + α_i_log, A_j )

        hmaps = ( (ε*A_grid.view(-1), x_i, xt_i), (ε*B_grid.view(-1), y_j, yt_j) )
    else : hmaps = (None, None)

    a_y, b_x = ε*A_j.view(-1), ε*B_i.view(-1)
    return a_y, b_x, hmaps


def sym_sink(α_i, x_i, y_j=None, β_j=None, p=1, eps=.1, nits=100, tol=1e-3, assume_convergence=False, heatmaps=None, **kwargs):

    ε = eps # Python supports Unicode. So fancy!
    # Sinkhorn loop ......................................................................

    α_i_log = α_i.log()
    A_i     = torch.zeros_like(α_i)
    S_x, _  = Sinkhorn_ops(p, ε, x_i, x_i) # Sinkhorn operator from x_i to x_i (divided by ε, as it's slightly cheaper...)
    
    # if we assume convergence, we can skip all the "save computational history" stuff
    torch.set_grad_enabled(not assume_convergence)

    for i in range(int(nits-1)) :
        A_i = 0.5 * (A_i + S_x(A_i + α_i_log) ) # a(x)/ε = .5*(a(x)/ε + Smin_ε,y~α [ C(x,y) - a(y) ] / ε)
        
    
    # Fancy display, in the background... 
    if heatmaps == True :
        global grid
        grid = grid.type_as(x_i)
        S2_x, _ = Sinkhorn_ops(p, ε, x_i, grid)
        A_grid = S2_x( A_i + α_i_log ) # a(y)/ε = Smin_ε,x~α [ C(x,y) - a(x) ]  / ε
        a_grid = ε*A_grid.view(-1)

    a_x = ε * S_x( A_i + α_i_log ).view(-1) # a(x) = Smin_e,z~α [ C(x,z) - a(z) ]

    if y_j is None : 
        return None, a_x, None
    else : # extrapolate "a" to the point cloud "y_j"
        S2_x, _ = Sinkhorn_ops(p, ε, x_i, y_j) # Sinkhorn operator from x_i to y_j (divided by ε...)
        A_y = S2_x( A_i + α_i_log ) # a(z) = Smin_e,z~α [ C(y,z) - a(z) ]
        return ε * A_y.view(-1), a_x, a_grid


#######################################################################################################################
# Derived Functionals .....................................................................
#######################################################################################################################

def  regularized_ot( α, x, β, y, **params): # OT_ε
    a_y, b_x, hmaps = sink( α, x, β, y, **params)
    cost = scal(α, b_x) + scal(β, a_y)

    return cost if (params.get("heatmaps", None) is None) else (cost, HeatmapsSprings(*hmaps)) 
        

def hausdorff_divergence(α, x, β, y, **params): # H_ε
    a_y, a_x, a_grid = sym_sink( α, x, y, **params)
    b_x, b_y, b_grid = sym_sink( β, y, x, **params)
    cost = .5 * ( scal( α, b_x - a_x ) + scal( β, a_y - b_y ) )

    if params.get("heatmaps", None) == True :
        ε = params["eps"]
        Rx, Ry = Barycenters_ops(params["p"], ε, x, y)
        Px, Py = Projection_ops(params["p"], ε, x, y)
        xt = x + Ry( ε*b_y.view(-1,1) + β.log(), 0*b_x.view(-1,1) ) \
                /Py( ε*b_y.view(-1,1) + β.log(), 0*b_x.view(-1,1) )
        yt = y + Rx( ε*a_x.view(-1,1) + α.log(), 0*a_y.view(-1,1) ) \
                /Px( ε*a_x.view(-1,1) + α.log(), 0*a_y.view(-1,1) )

        springs_a = a_grid, x, xt
        springs_b = b_grid, y, yt

    else : springs_a, springs_b = None, None

    return cost if (params.get("heatmaps", None) is None) else (cost, HeatmapsSprings(springs_a,springs_b)) 


def  sinkhorn_divergence(α, x, β, y, **params): # S_ε
    a_y, b_x, hmaps =     sink( α, x, β, y, **params)
    _,   a_x, _     = sym_sink( α, x,       **params )
    _,   b_y, _     = sym_sink( β, y,       **params )
    cost =  scal( α, b_x - a_x ) + scal( β, a_y - b_y )

    return cost if (params.get("heatmaps", None) is None) else (cost, HeatmapsSprings(*hmaps)) 





