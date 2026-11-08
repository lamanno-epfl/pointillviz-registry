# Seurat niches

Answers the question: **Which neighbourhoods of cell types does the tissue divide into?**

The tool runs `BuildNicheAssay`, the niche analysis of [Seurat](https://satijalab.org/seurat/), with the authors' own R package and the settings of their [Xenium vignette](https://satijalab.org/seurat/articles/seurat5_spatial_vignette_2). Only cell types and positions enter the analysis, not gene expression.

For each cell *i*, the tool finds its *k* nearest cells and counts how many of them belong to each cell type *t*, giving *n(i,t)*. These counts are then scaled for each cell type across all cells:

$$
z(i,t) = \frac{n(i,t) − n̄(t)}{s(t)}
$$

Here, *n̄(t)* and *s(t)* are the mean and standard deviation of *n(i,t)* over all cells. A positive *z(i,t)* therefore means that cell *i* has more type-*t* neighbours than the average cell. Finally, k-means groups the cells into the chosen number of niches, so that cells within one niche have similar neighbourhood compositions; the best of 30 random starts is kept.

**Map.** Each cell is colored by its niche. Niches are numbered N1, N2, … from the largest to the smallest; the numbers carry no other meaning.

**Matrix.** Rows are niches and columns are cell types, ordered by abundance. Each square shows the percentage of the niche's cells that belong to that cell type.

**Table.** Each row represents one niche. `Cells` and `% of cells` give its size, and `Main cell types` lists its three most common cell types with their shares. A download lists the niche of every cell.

The number of niches is a choice, not a finding: k-means returns as many as requested. With several sections, each cell's neighbours come from its own section, and all sections share one set of niches.
