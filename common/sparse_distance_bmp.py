
import torch
from torch.autograd import grad
from divergences    import kernel_divergence, regularized_ot, hausdorff_divergence, sinkhorn_divergence

def extract_point_cloud(I, affine) :
    """Bitmap to point cloud."""

    # Threshold, to extract the relevant indices ---------------------------------------
    ind = (I > .001).nonzero()

    # Extract the weights --------------------------------------------------------------
    D = len(I.shape)
    if   D == 2 : α_i = I[ind[:,0], ind[:,1]]
    elif D == 3 : α_i = I[ind[:,0], ind[:,1], ind[:,2]]
    else : raise NotImplementedError()

    α_i = α_i * affine[0,0] * affine[1,1] # Lazy approximation of the determinant...
    # If we normalize the measures, it doesn't matter anyway.

    # Don't forget the changes of coordinates! -----------------------------------------
    M   = affine[:D,:D] ; off = affine[:D,D]
    x_i = ind.float() @ M.t() + off

    return ind, α_i.view(-1,1), x_i

def sparse_distance_bmp(params, A, B, affine_A, affine_B, normalize=True, info=False, action="measure") :
    """
    Takes as input two torch bitmaps (Tensors). 
    Returns a cost and a gradient, encoded as a vector bitmap.

    Args :
        - A and B : two torch bitmaps (Tensors) of dimension D.
        - affine_A and affine_B : two matrices of size (D+1,D+1) (Tensors).
    """
    D = len(A.shape) # dimension of the ambient space, =2 for slices or =3 for volumes

    ind_A, α_i, x_i = extract_point_cloud(A, affine_A)
    ind_B, β_j, y_j = extract_point_cloud(B, affine_B)

    if normalize :
        α_i = α_i / α_i.sum()
        β_j = β_j / β_j.sum()

    x_i.requires_grad = True
    if action == "image" :
        α_i.requires_grad = True

    # Compute the distance between the *measures* A and B ------------------------------
    print("{:,}-by-{:,} KP: ".format(len(x_i), len(y_j)), end='')

    routines = { 
        "kernel"         : kernel_divergence,
        "regularized_ot" : regularized_ot,
        "hausdorff"      : hausdorff_divergence,
        "sinkhorn"       : sinkhorn_divergence, 
    }

    routine = routines[ params.get("formula", "hausdorff") ]
    params["heatmaps"] = info

    cost, heatmaps = routine( α_i,x_i, β_j,y_j, **params )

    if action == "image" :
        grad_a, grad_x = grad( cost, [α_i, x_i] ) # gradient wrt the voxels' positions and weights
    elif action == "measure" :
        grad_x = grad( cost, [x_i] )[0] # gradient wrt the voxels' positions

    # Point cloud to bitmap (grad_x) ---------------------------------------------------
    tensor   = torch.cuda.FloatTensor if A.is_cuda else torch.FloatTensor 
    # Using torch.zero(...).dtype(cuda.FloatTensor) would be inefficient...
    # Let's directly make a "malloc", before zero-ing in place
    grad_A = tensor( *(tuple(A.shape) + (D,))  )
    grad_A.zero_()

    if action == "measure" :
        if   D == 2 : grad_A[ind_A[:,0],ind_A[:,1],            :] = grad_x[:,:] 
        elif D == 3 : grad_A[ind_A[:,0],ind_A[:,1],ind_A[:,2], :] = grad_x[:,:]
        else :        raise NotImplementedError()

    elif action == "image" :
        if   D == 2 :
            if True :
                dim_0 = affine_A[0,0] ; print(dim_0)
                grad_A[ind_A[:,0]  ,ind_A[:,1]  , :] += .25 * dim_0 * grad_x[:,:]
                grad_A[ind_A[:,0]+1,ind_A[:,1]  , :] += .25 * dim_0 * grad_x[:,:]
                grad_A[ind_A[:,0]  ,ind_A[:,1]+1, :] += .25 * dim_0 * grad_x[:,:]
                grad_A[ind_A[:,0]+1,ind_A[:,1]+1, :] += .25 * dim_0 * grad_x[:,:]

            grad_a = grad_a[:] * alpha_i[:]
            grad_A[ind_A[:,0]  ,ind_A[:,1]  , 0] -= .5*grad_a[:]
            grad_A[ind_A[:,0]+1,ind_A[:,1]  , 0] += .5*grad_a[:]
            grad_A[ind_A[:,0]  ,ind_A[:,1]+1, 0] -= .5*grad_a[:]
            grad_A[ind_A[:,0]+1,ind_A[:,1]+1, 0] += .5*grad_a[:]

            grad_A[ind_A[:,0]  ,ind_A[:,1]  , 1] -= .5*grad_a[:]
            grad_A[ind_A[:,0]  ,ind_A[:,1]+1, 1] += .5*grad_a[:]
            grad_A[ind_A[:,0]+1,ind_A[:,1]  , 1] -= .5*grad_a[:]
            grad_A[ind_A[:,0]+1,ind_A[:,1]+1, 1] += .5*grad_a[:]
 
            if False :
                grad_A[ind_A[:,0]  ,ind_A[:,1]  , 0] = grad_a[:]
                grad_A[ind_A[:,0]  ,ind_A[:,1]  , 1] = grad_a[:]
            
    # N.B.: we return "PLUS gradient", i.e. "MINUS a descent direction".
    return cost, grad_A.detach(), heatmaps

