
#######################################################################################
#         Load, Sample, Display density from png source                               #
#######################################################################################

import numpy as np
import torch
from random import choices
from scipy import misc
from matplotlib import pyplot as plt


def load_image(fname) :
    img = misc.imread(fname, flatten = True) # Grayscale
    img = (img[::-1, :])  / 255.
    #img = np.swapaxes(img, 0,1 )
    return 1 - img

def draw_samples(fname, n, dtype=torch.FloatTensor) :
    A = load_image(fname)
    xg, yg = np.meshgrid( np.linspace(0,1,A.shape[0]), np.linspace(0,1,A.shape[1]) )
    
    grid = list( zip(xg.ravel(), yg.ravel()) )
    dens = A.ravel() / A.sum()
    dots = np.array( choices(grid, dens, k=n ) )
    dots += .005 * np.random.standard_normal(dots.shape)

    weights = torch.ones(n,1).type(dtype) / n
    return weights, torch.from_numpy(dots).type(dtype)

def display_samples(ax, x, color, x_grad=None) :
    x_ = x.data.cpu().numpy()
    ax.scatter( x_[:,0], x_[:,1], 16 * 500 / len(x_), color )

    if x_grad is not None :
        g_ = -x_grad.data.cpu().numpy()
        ax.quiver( x_[:,0], x_[:,1], g_[:,0], g_[:,1], 
                    scale = .05/ len(x_), scale_units="dots", color="#5CBF3A", 
                    zorder=3, width=0.0025)

