#########################################################################################################################
#  Heatmaps, displayed in the background  ---------------------------------------------------------------------------
#########################################################################################################################

import torch
import numpy as np

# To keep it simple, we hardcode a uniform grid on the unit square through a global variable:
xmin,xmax,ymin,ymax,res  = coords = 0,1,0,1,100
ticks_x = np.linspace( xmin, xmax, res + 1)[:-1] + 1/(2*res) 
ticks_y = np.linspace( ymin, ymax, res + 1)[:-1] + 1/(2*res) 
X,Y     = np.meshgrid( ticks_x, ticks_y )
grid    = torch.from_numpy(np.vstack( (X.ravel(), Y.ravel()) ).T).contiguous()

class Heatmaps :
    def __init__(self, a=None, b=None) :
        # reshape as a "background" image
        self.a = None if a is None else a.view(res,res).data.cpu().numpy() 
        self.b = None if b is None else b.view(res,res).data.cpu().numpy()

    def plot(self, axis) :
        def contour_plot(img, color, nlines=15, zero=False) :
            levels = np.linspace(np.amin( img ), np.amax( img ), nlines)
            axis.contour(img, origin='lower', linewidths = 1., colors = color,
                        levels = levels, extent=coords[0:4]) 
            if zero :
                try : # Bold line for the zero contour line; throws a warning if no "0" is found
                    axis.contour(img, origin='lower', linewidths = 2., colors = color,
                                levels = (0.), extent=coords[0:4]) 
                except : pass

        if self.b is None :
            contour_plot(self.a, "#C8DFF9", nlines=31, zero=True)
        else :
            contour_plot(self.a, "#E2C5C5")
            contour_plot(self.b, "#C8DFF9")


from   matplotlib.collections  import LineCollection


class HeatmapsSprings :
    def __init__(self, springs_a=None, springs_b=None) :
        # reshape as a "background" image
        self.a = None if springs_a is None else springs_a[0].view(res,res).data.cpu().numpy() 
        self.b = None if springs_b is None else springs_b[0].view(res,res).data.cpu().numpy()

        self.x_i  = None if springs_a is None else springs_a[1].data.cpu().numpy() 
        self.xt_i = None if springs_a is None else springs_a[2].data.cpu().numpy()
        self.y_j  = None if springs_b is None else springs_b[1].data.cpu().numpy() 
        self.yt_j = None if springs_b is None else springs_b[2].data.cpu().numpy()


    def plot(self, axis) :
        def contour_plot(img, color, nlines=15, zero=False) :
            levels = np.linspace(np.amin( img ), np.amax( img ), nlines)
            axis.contour(img, origin='lower', linewidths = 1., colors = color,
                        levels = levels, extent=coords[0:4]) 
            if zero :
                try : # Bold line for the zero contour line; throws a warning if no "0" is found
                    axis.contour(img, origin='lower', linewidths = 2., colors = color,
                                levels = (0.), extent=coords[0:4]) 
                except : pass

        contour_plot(self.a, "#E2C5C5")
        contour_plot(self.b, "#C8DFF9")

        # Springs
        springs_a = [ [s_i,t_i] for (s_i,t_i) in zip(self.x_i,self.xt_i)]
        springs_b = [ [s_j,t_j] for (s_j,t_j) in zip(self.y_j,self.yt_j)]
        seg_colors_a = [ (.8, .4, .4, .05) ] * len(self.x_i)
        seg_colors_b = [ (.4, .4, .8, .05) ] * len(self.y_j)
        
        line_segments = LineCollection(springs_b, linewidths=(1,), 
                                       colors=seg_colors_b, linestyle='solid')
        axis.add_collection(line_segments)
        
        line_segments = LineCollection(springs_a, linewidths=(1,), 
                                       colors=seg_colors_a, linestyle='solid')
        axis.add_collection(line_segments)

