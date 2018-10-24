# Global divergences

This repository provides efficient implementations of
Maximum Mean Discrepancies (aka. kernel norms),
Hausdorff and Sinkhorn divergences between sampled measures.
Thanks to the [KeOps library](https://www.kernel-operations.io),
our routines scale up to batches of 1,000,000 samples, **without memory overflows**.

Information on the subject is available in our papers:

- *Global divergences between measures : from Hausdorff distance to Optimal Transport*,
  Jean Feydy, Alain Trouvé, [ShapeMI2018](https://shapemi.github.io/); [HAL version](https://hal.archives-ouvertes.fr/hal-01827184/).

- *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences*,
  Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, Gabriel Peyré; [ArXiv](https://arxiv.org/abs/1810.08278).
  
- *Sinkhorn entropies and divergences*,
  Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, Gabriel Peyré; (our long reference paper, available soon).

First and foremost, this repo is about providing a reference implementation of Sinkhorn-related divergences. In [`/common/`](./common), you will find
a [simple](./common/sinkhorn_balanced_simple.py) and
an [efficient](./common/sinkhorn_balanced.py) implementation of
the Sinkhorn algorithm.
The folder [`global_divergences_ShapeMI2018`](./global_divergences_ShapeMI2018)
will let you reproduce the figures of our ShapeMI paper (Miccai 2018 workshop),
while [`sinkhorn_entropies`](./sinkhorn_entropies) contains those
of our reference theoretical article.

Please note that to run some of our demos, you will need to install
both [pytorch](https://pytorch.org/) and [KeOps](https://www.kernel-operations.io). No worries: `pip install pykeops` should do the trick.

