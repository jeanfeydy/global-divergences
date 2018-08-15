#-------------------------------------------------------
#            Code used to generate Fig. 2
#-------------------------------------------------------


import numpy as np

import torch
from pykeops.torch import Kernel, kernel_product

s2v = lambda x : torch.tensor([x])

Nt, Nx = 501, 201
t   = torch.linspace( 0, 1, Nt ).view(-1,1)
x_i = torch.linspace( .2, .35, Nx).view(-1,1)
y_j = torch.linspace( .65, .8, Nx).view(-1,1)
mu_i = .15 * torch.ones( Nx ).view(-1,1) / Nx
nu_j = .15 * torch.ones( Nx ).view(-1,1) / Nx


kernels =  {
    "gaussian" : { 
        "id"         : Kernel("gaussian(x,y)"),
        "gamma"      : s2v( 1 / .1**2 ),
    },
    "laplacian" : { 
        "id"         : Kernel("laplacian(x,y)"),
        "gamma"      : s2v( 1 / .1**2 ),
    },
    "energy_distance" : { 
        "id"         : Kernel("-distance(x,y)"),
        "gamma"      : s2v( 1. ),
    },
}

fs = [ t ]
for name, kernel in kernels.items() :
    f = kernel_product( kernel, t, x_i, mu_i) \
      - kernel_product( kernel, t, y_j, nu_j)
    fs.append(f)


header = "t gaussian laplacian energy_distance"
lines  = [ f.view(-1).data.cpu().numpy() for f in fs ]

data = np.stack(lines).T
np.savetxt("output/graphs/kernel_1D.csv", data, fmt='%-9.5f', header=header, comments = "")
