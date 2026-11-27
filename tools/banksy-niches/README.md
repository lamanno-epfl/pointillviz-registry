# BANKSY niches

Answers the question: **Which tissue domains, or niches, does the tissue divide into, based on what each cell and its neighbours express?**

The tool runs [BANKSY](https://github.com/prabhakarlab/Banksy_py) with the authors' own Python package, `pybanksy`, and their default settings. BANKSY describes each cell by its own expression together with that of its spatial neighbours, and groups cells with similar descriptions into niches.

The counts are first normalized and reduced to the most variable genes, which gives each cell *i* its own expression *x(i)*. The tool then averages the expression of the cell's *k* nearest neighbours *N(i)*, using weights *w(i,j)* that sum to 1 and, by default, decrease with distance:

$$
m(i) = \sum_{j∈N(i)} w(i,j) x(j)
$$

By default, a third block is added: the azimuthal Gabor filter (AGF) *g(i)*, which weights neighbouring cells by their direction around cell *i*. It therefore responds where expression changes from one side of the neighbourhood to the other, as at a tissue border. Each block is z-scored gene by gene, and the blocks are joined using the neighbourhood weight *λ*:

$$
b(i) = \paren{√(1−λ) x(i), √(2λ/3) m(i), √(λ/3) g(i)}
$$

Without the AGF, *m(i)* receives the full weight *√λ*. With *λ* = **0.8**, the neighbourhood dominates and the niches follow tissue domains; with *λ* = **0.2**, they resemble spatially informed cell types. The joined profiles are reduced to their principal components and clustered with the Leiden algorithm, and each cluster forms one niche.

**Map.** Each cell is colored by its niche. Niches are numbered N1, N2, … from the largest to the smallest; the numbers carry no other meaning.

**Matrix.** Rows are niches and columns are cell types, ordered by abundance. Each square shows the percentage of the niche's labelled cells that belong to that cell type. Without a cell-type annotation, the columns show the niches' top marker genes instead, colored by their marker z-scores.

**Table.** Each row represents one niche. `Cells` and `% of cells` give its size. `Marker genes` lists the five genes whose mean expression in the niche has the highest z-score relative to all cells, while `Main cell types` gives its three most common cell types with their shares. A download lists the niche of every cell.

With several sections, each cell's neighbours come from its own section, and all sections share one set of niches.
