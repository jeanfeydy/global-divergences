#-------------------------------------------------------
#            Code used to generate Fig. 5
#-------------------------------------------------------

import numpy as np
import torch

s2v = lambda x : torch.tensor([x])

Nt  = 501
t   = torch.linspace( 0, 1, Nt ).view(-1,1)
x_1 = .25
x_2 = .75

def lse( a, b ) :
    c = torch.max(a,b)
    return c + torch.log( (a-c).exp() + (b-c).exp())


fs = [ t ]
fs.append( .5*( (t-x_1).abs() + (t-x_2).abs() ) )
for eps in [ 5., .5, .1 ] :
    fs.append( - eps * lse( - (t-x_1).abs()/eps - np.log(2), - (t-x_2).abs()/eps - np.log(2) )  )

fs.append( torch.min( (t-x_1).abs(), (t-x_2).abs() ) )


header = "t inf large medium small zero"
lines  = [ f.view(-1).data.cpu().numpy() for f in fs ]

data = np.stack(lines).T
np.savetxt("output/graphs/softmin.csv", data, fmt='%-9.5f', header=header, comments = "")
