# Seurat niches

Answers the question: **Which neighbourhoods of cell types does the tissue divide into?**

The tool runs the niche analysis of [Seurat](https://satijalab.org/seurat/) (Hao et al., *Nature Biotechnology* 2024) with the authors' own R package, **imported from conda-forge as they ship it** (`r-seurat` 5.5.1). It calls `BuildNicheAssay` as their [Xenium vignette](https://satijalab.org/seurat/articles/seurat5_spatial_vignette_2) does, with their values, and holds no copy of their method. The function:

1. finds each cell's nearest cells;
2. counts the cell types among them and scales those counts;
3. groups cells with similar surroundings into niches with k-means, keeping the best of 30 random starts.

The authors credit Goltsev et al. (*Cell* 2018) and He et al. (*Nature Biotechnology* 2022) as its inspiration. Only cell types and positions enter, not expression.

Called directly with the real counts in the object, with no code of this tool, the function gives the same niches, cell for cell, on a Xenium 5K mouse colon section of 135,231 cells. Asked for 8 niches there, it separates smooth muscle, the layers of enterocytes and goblet cells, and a small niche of lymphoid aggregates (0.7% of cells: dendritic cells and ILC3s).

**Map.** Every cell painted by its niche, numbered N1, N2, … from the largest.

**Matrix.** Each niche's composition: the share of its cells in each cell type.

**Table.** Each niche's size and main cell types, with a download of every cell's niche.

**Several sections.** The function takes one set of coordinates, so the sections are set apart in space before it is called. A cell's neighbours then come from its own section, and all sections share one set of niches. A section with no more cells than a neighbourhood holds is left out, and the summary says so. Isolate a section, or select cells, to run on those alone.

**Cells left out.** Those with no cell type, as in the vignette, and those with no coordinates. The summary counts them.

**What the package does as shipped.** The tool leaves these as they are:

- The number of niches is a choice, not a finding: k-means makes as many as asked, and no niche carries a p-value.
- Neighbours come from an approximate search (Annoy), the default of `FindNeighbors`.
- Each k-means start stops at R's default of 10 iterations, and the summary says when that cut one short. On the section above, 100 iterations changed no cell's niche.

**Cost.** Memory and time grow with cells × cell types: 135,231 cells with 73 cell types take about 2 minutes and 2 GB, and 708,889 cells with 327 about 24 GB. If the memory is not there, the tool says so before it starts. It needs 2D coordinates.

`Seurat` is under the MIT licence, and so is this tool.
