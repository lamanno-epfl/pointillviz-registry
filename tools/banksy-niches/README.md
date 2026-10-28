# BANKSY niches

Answers the question: **Which tissue domains, or niches, does a section divide into, judged by what each cell and its neighbours express?**

BANKSY (Singhal, Chou, Lee et al., *Nature Genetics* 2024) describes every cell by its own expression together with the expression around it, and clusters the cells on that joint description. With most of the weight on the neighbourhood, the clusters are tissue domains; with little, they are cell types refined by their surroundings.

## What you get

- **Map.** Every cell painted by its niche, one colour per niche. Niches are numbered N1, N2, … from the largest, and the numbers carry no other order. Cells without spatial neighbours, or outside a selection, are left unpainted.
- **Matrix.** Each niche's cell-type composition when the dataset has cell types; its marker genes otherwise.
- **Table.** Each niche's size, its five strongest marker genes and its three main cell types.
- **Download.** A CSV (gzip) of every cell's niche, when it fits in the result.

On a Xenium 5K mouse colon section of 135,000 cells, the defaults give 8 niches that follow the structure of the gut wall: the muscle layers, the lamina propria, crypt and surface epithelium, and a small niche of lymphoid aggregates (ILC3s and dendritic cells, with Cxcl13 and Ltb).

## Parameters

- **Clustering.** *Leiden*, the authors' default, finds as many niches as the resolution calls for. *k-means* divides the cells into exactly the number of niches you ask for.
- **Resolution (Leiden).** Higher values give more, smaller niches; lower values give fewer, larger ones.
- **Niches (k-means).** How many niches k-means makes.
- **Neighbourhood weight (λ).** 0.8 finds tissue domains, 0.2 spatially informed cell types, and 0 ignores space.
- **Spatial neighbours (k).** How many nearest cells make a neighbourhood. The gradient features use twice as many.
- **Include gradients (AGF).** Adds BANKSY's azimuthal Gabor filter, which responds where expression changes across a neighbourhood, as at a border.
- **Variable genes.** How many of the most variable genes to use. More genes take longer.
- **Principal components.** How many components of the joint matrix the clustering sees.
- **Smooth labels.** Gives a cell the niche held by more than half of itself and its 15 nearest cells, repeated until nothing changes (BANKSY's smoothLabels).

## Method

For each cell, three blocks over the most variable genes, ranked by dispersion as in scanpy's `flavor="seurat"`:

- **own**: the cell's library-size-normalised expression;
- **neighbourhood mean**: a Gaussian-weighted average over its *k* nearest cells, the kernel scaled by the cell's median neighbour distance;
- **gradient (AGF)**: the azimuthal Gabor filter over its 2*k* nearest cells. It is large where expression changes across the neighbourhood.

Each column is z-scored, the blocks are weighted by √(1−λ), √(2λ/3) and √(λ/3) (√(1−λ) and √λ without the gradient), and the joint matrix is reduced to its top principal components. The components are then clustered:

- **Leiden**, as the authors' Python package runs it. Each cell is linked to its 50 nearest cells in component space, with weight 1/(1 + distance). A link is kept when the two cells share five or more of those neighbours, or when it is one of the three links the cell shares the most neighbours over. Leiden (RBConfiguration quality, seed 1234, repeated until no cell moves) partitions this graph.
- **k-means**, with k-means++ seeding and the best of four fixed starts. The authors' R package offers it too (`clusterBanksy(algo = "kmeans")`).

The joint matrix is streamed in chunks rather than stored, so memory stays near the size of its Gram matrix, and the chunks are built on up to 16 processors.

This is a reimplementation in NumPy and SciPy, with python-igraph and leidenalg for Leiden. It takes no code from the authors' packages. On that section, its three feature blocks match `pybanksy` 1.3.5 to within 5 × 10⁻⁸, and its principal components are exact, where `pybanksy`'s default randomized PCA is unseeded and changes slightly from run to run.

## Citation

Singhal V, Chou N, Lee J, et al. BANKSY unifies cell typing and tissue domain segmentation for scalable spatial omics data analysis. *Nature Genetics* 56, 431–441 (2024).
