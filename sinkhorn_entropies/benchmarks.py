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

from pykeops.torch import generic_sum, generic_logsumexp


MAXTIME = 300 # 5mn
REDTIME = 10  # Decrease the number of loops if computations take longer than 10s...
D  = 3 # Let's do this in 3D
NS = [10, 20, 50, 
      100, 200, 500, 
      1000, 2000, 5000, 
      10000, 20000, 50000, 
      100000, 200000, 500000,
      1000000]



def benchmark(bench_name, N, dev, backend, loops = 10, enable_GC=True, fidelity=None) :

    importlib.reload(torch)

    device = torch.device(dev)
    x_i  = torch.randn(N, D, dtype=torch.float32, device=device, requires_grad=True)
    y_j  = torch.randn(N, D, dtype=torch.float32, device=device)
    α_i  = torch.randn(N, 1, dtype=torch.float32, device=device)
    β_j  = torch.randn(N, 1, dtype=torch.float32, device=device)

    α_i = α_i.abs()       ; β_j = β_j.abs()
    α_i = α_i / α_i.sum() ; β_j = β_j / β_j.sum()

    s2v = lambda x : torch.tensor([x], dtype=torch.float32, device=device)

    def scal( α, f ) :
        return torch.dot( α.view(-1), f.view(-1) )

    if bench_name == "energy_distance" :
        keops_conv = generic_sum( "Sqrt(SqDist(Xi,Yj))* Bj", "out_i = Vx(1)", # Formula, output...
            # and input variables : x_i, y_j, β_j, given with their respective dimensions
            "Xi = Vx({})".format(D), "Yj = Vy({})".format(D), "Bj = Vy(1)")

        def vanilla_conv(x,y,β) :
            XmY2 = ( (x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
            K =  XmY2.sqrt()
            return K @ β

        def bench(α,x,β,y) :
            if   backend == "GPU_1D" :
                conv = keops_conv
            elif backend == "pytorch" :
                conv = vanilla_conv
            cost = scal(α, conv(x,y,β) - .5*conv(x,x,α) ) - .5*scal(β, conv(y,y,β))
            cost.backward()
            return cost

        code = '_ = bench(α_i,x_i,β_j,y_j)'
        task = "Energy Distances"

    if bench_name == "LogSumExp" :
        keops_lse = generic_logsumexp( "Sqrt(SqDist(Xi,Yj))", "out_i = Vx(1)", # Formula, output...
            # and input variables : x_i, y_j, β_j, given with their respective dimensions
            "Xi = Vx({})".format(D), "Yj = Vy({})".format(D))

        def lse( v_ij ):
            """[lse(v_ij)]_i = log sum_j exp(v_ij), with numerical accuracy."""
            V_i = torch.max(v_ij, 1)[0].view(-1,1)
            return V_i + (v_ij - V_i).exp().sum(1).log().view(-1,1)

        def vanilla_lse(x,y) :
            XmY2 = ( (x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
            K =  XmY2.sqrt()
            return lse( K )

        def bench(x,y) :
            if   backend == "GPU_1D" :
                return keops_lse(x,y)
            elif backend == "pytorch" :
                return vanilla_lse(x,y)
            else :
                raise NotImplementedError()
        code = '_ = bench(x_i,y_j)'
        task = "LSEs"

    elif bench_name == "fidelities" :

        from divergences import kernel_divergence, regularized_ot, hausdorff_divergence, sinkhorn_divergence

        if fidelity == "energy_distance" :
            params = ("energy", None)
            code = "c = kernel_divergence(α_i,x_i, β_j,y_j, k=params ) ; c.backward()"

        elif fidelity == "hausdorff" :
            params = {
                "p"    : 1,
                "eps"  : .1,
                "nits" : 3,
                "tol"  : 0.,
            }
            code = "c = hausdorff_divergence(α_i,x_i, β_j,y_j, **params ) ; c.backward()"

        elif fidelity == "sinkhorn" :
            params = {
                "p"    : 1,
                "eps"  : .1,
                "nits" : (20,3),
                "assume_convergence" : True, # This is true in practice, and lets us win a x2 factor
                "tol"  : 0.,
            }
            code = "c = sinkhorn_divergence(α_i,x_i, β_j,y_j, **params ) ; c.backward()"

        elif fidelity == "sinkhorn_nocv" :
            params = {
                "p"    : 1,
                "eps"  : .1,
                "nits" : (20,3),
                "assume_convergence" : False,
                "tol"  : 0.,
            }
            code = "c = sinkhorn_divergence(α_i,x_i, β_j,y_j, **params ) ; c.backward()"

        task = "fidelities"

    exec(code, locals())
    import gc
    GC = 'gc.enable();' if enable_GC else 'pass;'
    print("{:3} NxN {}, with N ={:7}: {:3}x".format(loops, task, N, loops), end="")

    exec( code, locals() ) # Warmup run

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
            or (nloops * elapsed > REDTIME/10 and len(Nloops) > 0 ) : 
                nloops = Nloops.pop(0)

    except RuntimeError :
        print("**\nMemory overflow !")
    except IndexError :
        print("**\nToo slow !")
    
    return times + (len(NS)-len(times)) * [np.nan]

def full_bench(bench_name) :
    print("Benchmarking : {} ==========================================".format(bench_name))
    lines  = [ NS ]

    if bench_name in ["energy_distance", "LogSumExp"] :
        header = "Npoints GPU_1D pytorch_cpu pytorch_gpu"
        #lines.append( bench_config(bench_name, "cpu",  "CPU") )
        lines.append( bench_config(bench_name, "cuda", "GPU_1D") )
        lines.append( bench_config(bench_name, "cpu",  "pytorch") )
        lines.append( bench_config(bench_name, "cuda", "pytorch") )

    elif bench_name == "fidelities" :
        fidelities = ["energy_distance", "hausdorff", "sinkhorn", "sinkhorn_nocv"]
        header = "Npoints " + " ".join(fidelities)
        for s in fidelities :
            lines.append( bench_config(bench_name, "cuda", "GPU_1D", fidelity=s) )

    benchs = np.array(lines).T
    np.savetxt("output/benchmarks/benchmark_"+bench_name+".csv", benchs, 
               fmt='%-9.5f', header=header, comments='')

if __name__ == "__main__" :
    #full_bench("energy_distance")
    #full_bench("LogSumExp")
    full_bench("fidelities")
