#-------------------------------------------------------
#            Code used to generate Fig. 11
#-------------------------------------------------------


import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'common')


import numpy as np
import time, timeit

import importlib
import torch
assert torch.cuda.is_available(), "No point running this bench without a GPU !"

from pykeops.torch import Kernel


MAXTIME = 10
D  = 3 # Let's do this in 3D
NS = [10, 20, 50, 
      100, 200, 500, 
      1000, 2000, 5000, 
      10000, 20000, 50000, 
      100000, 200000, 500000,
      1000000, 2000000, 5000000,
      10000000]


def benchmark(bench_name, N, dev, backend, loops = 10, enable_GC=True, fidelity=None) :

    importlib.reload(torch)

    device = torch.device(dev)
    x_i  = torch.randn(N, D, dtype=torch.float32, device=device, requires_grad=True)
    y_j  = torch.randn(N, D, dtype=torch.float32, device=device)
    mu_i = torch.randn(N, 1, dtype=torch.float32, device=device)
    nu_j = torch.randn(N, 1, dtype=torch.float32, device=device)

    mu_i = mu_i.abs()        ; nu_j = nu_j.abs()
    mu_i = mu_i / mu_i.sum() ; nu_j = nu_j / nu_j.sum()

    s2v = lambda x : torch.tensor([x], dtype=torch.float32, device=device)

    if bench_name == "gaussian_conv" :
        k = { "id"         : Kernel("gaussian(x,y)"),
              "gamma"      : s2v( .25 ),
              "backend"    : backend,                 }

        from pykeops.torch import kernel_product

        _ = kernel_product(k, x_i, y_j, nu_j)
        import gc
        GC = 'gc.enable();' if enable_GC else 'pass;'
        print("{:3} NxN-gaussian-convs, with N ={:7}: {:3}x".format(loops, N, loops), end="")

        elapsed = timeit.Timer('_ = kernel_product(k,x_i,y_j,nu_j)', GC,  
                                        globals = locals(), timer = time.time).timeit(loops)

    elif bench_name == "fidelities" :

        from divergences import kernel_divergence, regularized_ot, hausdorff_divergence, sinkhorn_divergence

        if fidelity == "energy_distance" :
            params = ("energy", None)
            c = kernel_divergence(mu_i,x_i, nu_j,y_j, k=params ) ; c.backward()
            code = "c = kernel_divergence(mu_i,x_i, nu_j,y_j, k=params ) ; c.backward()"

        elif fidelity == "gaussian_kernel" :
            params = ("gaussian", .25)
            c = kernel_divergence(mu_i,x_i, nu_j,y_j, k=params ) ; c.backward()
            code = "c = kernel_divergence(mu_i,x_i, nu_j,y_j, k=params ) ; c.backward()"

        elif fidelity == "log_kernel" :
            params = {
                "p"    : 1,
                "eps"  : .1,
                "nits" : 1,
                "tol"  : 0.,
            }
            c = hausdorff_divergence(mu_i,x_i, nu_j,y_j, **params ) ; c.backward()
            code = "c = hausdorff_divergence(mu_i,x_i, nu_j,y_j, **params ) ; c.backward()"

        elif fidelity == "hausdorff" :
            params = {
                "p"    : 1,
                "eps"  : .1,
                "nits" : 3,
                "tol"  : 0.,
            }
            c = hausdorff_divergence(mu_i,x_i, nu_j,y_j, **params ) ; c.backward()
            code = "c = hausdorff_divergence(mu_i,x_i, nu_j,y_j, **params ) ; c.backward()"

        elif fidelity == "sinkhorn" :
            params = {
                "p"    : 1,
                "eps"  : .1,
                "nits" : 20,
                "assume_convergence" : True, # This is true in practice, and lets us win a x2 factor
                "tol"  : 0.,
            }
            c = sinkhorn_divergence(mu_i,x_i, nu_j,y_j, **params ) ; c.backward()
            code = "c = sinkhorn_divergence(mu_i,x_i, nu_j,y_j, **params ) ; c.backward()"

        import gc
        GC = 'gc.enable();' if enable_GC else 'pass;'
        print("{:3} NxN fidelities, with N ={:7}: {:3}x".format(loops, N, loops), end="")

        elapsed = timeit.Timer(code, GC, globals = locals(), timer = time.time).timeit(loops)

    print("{:3.6f}s".format(elapsed/loops))
    return elapsed / loops

def bench_config(bench_name, dev, backend, fidelity=None) :
    if fidelity is None :
        print("Backend : {}, Device : {} -------------".format(backend,dev))
    else :
        print("Backend : {}, Device : {}, Fidelity : {} -------------".format(backend,dev, fidelity))

    times = []
    try :
        Nloops = [100, 10, 1]
        nloops = Nloops.pop(0)
        for n in NS :
            elapsed = benchmark(bench_name, n, dev, backend, loops=nloops, fidelity=fidelity)
            times.append( elapsed )
            if (nloops * elapsed > MAXTIME) \
            or (nloops * elapsed > MAXTIME/10 and len(Nloops) > 0 ) : 
                nloops = Nloops.pop(0)

    except RuntimeError :
        print("**\nMemory overflow !")
    except IndexError :
        print("**\nToo slow !")
    
    return times + (len(NS)-len(times)) * [np.nan]

def full_bench(bench_name) :
    print("Benchmarking : {} ==========================================".format(bench_name))
    lines  = [ NS ]

    if bench_name == "gaussian_conv" :
        header = "Npoints CPU GPU_1D pytorch_cpu pytorch_gpu"
        lines.append( bench_config(bench_name, "cpu",  "CPU") )
        lines.append( bench_config(bench_name, "cuda", "GPU_1D") )
        lines.append( bench_config(bench_name, "cpu",  "pytorch") )
        lines.append( bench_config(bench_name, "cuda", "pytorch") )

    elif bench_name == "fidelities" :
        fidelities = ["energy_distance", "gaussian_kernel", "log_kernel", "hausdorff", "sinkhorn"]
        header = "Npoints " + " ".join(fidelities)
        for s in fidelities :
            lines.append( bench_config(bench_name, "cuda", "GPU_1D", fidelity=s) )

    benchs = np.array(lines).T
    np.savetxt("output/benchmarks/benchmark_"+bench_name+".csv", benchs, fmt='%-9.5f', header=header)

if __name__ == "__main__" :
    #full_bench("gaussian_conv")
    full_bench("fidelities")
