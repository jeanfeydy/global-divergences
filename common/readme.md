# Common routines

In this folder, you will find the generic routines used
across our papers.
Most importantly, we provide three implementations
of the Sinkhorn loop and divergences:

- [`sinkhorn_balanced_simple`](./sinkhorn_balanced_simple.py),
  which is a straightforward pytorch implementation.
- [`sinkhorn_balanced`](./sinkhorn_balanced.py),
  which is our *reference implementation*.
  It supports both vanilla pytorch *and* pytorch+KeOps backends;
  it allows you to bypass the naive differentiation of the Sinkhorn loop;
  and it lets you display the influence fields as
  contour lines in the background.
- [`sinkhorn_balanced_visualization`](./sinkhorn_balanced_visualization.py),
  which lets you display the "springs" associated to a transport
  plan, on top of the influence fields.
  It also supports halved iterations of the Sinkhorn loop.

On top of the Sinkhorn algorithm, this folder also provides:

- An efficient implementation of MMD/kernel norms, in [`kernel_norm`](./kernel_norm.py).
- Convenient, high-level divergence functions between measures:
  [`divergences`](./divergences.py) for sampled measures
  and [`sparse_distance_bmp`](./sparse_distance_bmp.py) for
  densities supported on a 2D or 3D grid, encoded as bitmaps.
- Fancy heatmaps and springs visualizations, in [`display`](./display.py).


**Dimensions of the input variables.** 
Following the conventions that we detail in our papers,
these routines work on sampled measures α and β, encoded
as sums of Dirac masses

α = ∑_i α_i·δ_{x_i} ,   β = ∑_j β_j·δ_{y_j}.

Here, α_i, x_i, β_j, y_j are all torch Tensors of shapes
N-by-1, N-by-D, M-by-1 and M-by-D, where N, M are the number of samples
in both measures and D is the dimension of the ambient feature 
space - typically, D=2 or 3 in shape analysis.