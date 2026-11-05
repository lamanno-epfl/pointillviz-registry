# Smoothie

Answers the question: **Which genes share a spatial pattern in this tissue, and where does each group sit?**

The tool runs [Smoothie](https://caholdener01.github.io/Smoothie/) (Holdener & De Vlaminck, *Communications Biology* 2026) as its authors ship it. It installs the published package, `smoothie-st`, and calls its functions in the order of the authors' pipeline; it holds no copy of their method. The package:

1. **smooths** each gene over the tissue with a Gaussian of the chosen width, which fills in the gaps between sparse counts;
2. **correlates** every pair of smoothed genes across the tissue;
3. **links** two genes when their correlation *r* passes the cutoff *c*, with a weight that the power *p* pushes down for weak links:

$$
w = \left(\frac{r - c}{1 - c}\right)^{p}
$$

4. **clusters** that network into modules with Infomap. A gene with no link joins no module, and modules are numbered from the largest.

On a Xenium 5K section of the mouse maternal–fetal interface (235,803 cells, 5,106 genes), the defaults give 47 modules holding 2,428 genes. Among them are smooth muscle (Actg2, Myh11), yolk-sac endoderm (Apob, Afp), NK cells (Prf1, Gzmb), lymphatics (Ccl21a, Prox1), proliferation (Mki67, Top2a), the interferon response (Isg15, Ifit1), B cells (Cd79a, Cd19) and plasma cells (Igkc, Jchain).

## What you get

- **Table.** One row per module of two genes or more: its number, its number of genes, its mean margin and its most connected genes, first the strongest. A table cell holds 200 characters, so a large module ends with how many genes are left: they are all in the download. A gene's margin compares its average link weight inside its module, *A*, with that to the closest other module, *B*, as (*A* − *B*) / max(*A*, *B*): 1 for a gene linked only inside its module, 0 for one split evenly between two. A low mean margin marks a module that is not sharply set apart.
- **Figure.** One panel per module, drawn by the package and numbered as in the table. Each location's score is the sum of the module's smoothed genes, each scaled from 0 to 1 by its own highest value. Grey is low, red high and black highest, within each panel, so compare patterns between panels, not their strengths. A small module can look pale beside its few darkest spots.
- **Download.** `smoothie-modules.csv`: every gene in a module, with the package's measures of it in the network (degree, weighted degree, rescaled weighted degree, clustering coefficient and margin).
- **Summary.** The numbers of this run, the three largest modules and the authors' lower bound for the cutoff.

**The lower bound.** The package repeats the smoothing and the correlations with the positions shuffled, and gives the 99.9th percentile of the correlations it finds. The authors use it as the lowest sensible cutoff. The summary gives it for this run, and says so in bold when the cutoff is below it.

## Parameters

- **Smoothing width (microns).** The standard deviation of the Gaussian. The authors find 20 to 40 µm works best. Wider finds broader patterns and needs fewer cells. If more than 15% of the tissue has fewer than 10 cells within three widths, the package refuses, and the tool says which width would pass.
- **Correlation cutoff.** The correlation above which two genes are linked. Higher keeps fewer genes, in tighter modules; lower takes longer. With no pair above it, the tool refuses and names the highest correlation found.
- **Clustering power.** How strongly weak links are discounted when the network is clustered. Higher gives more, smaller modules. The authors find 1 to 6 reasonable and examine values up to 9.

## What is fixed

Everything else is what the authors' pipeline notebooks give for cell-sized data, and the package's own defaults:

- **Quality control and normalisation.** Cells with at least 50 transcripts, genes with at least 100 counts in at least 10 cells; counts per thousand, then log1p.
- **Where it smooths.** The authors smooth subcellular to cell-sized data on a hexagonal grid, one width apart, and binned data at each spot. Their table starts binned data at 20 µm, so the tool smooths on the grid when cells or spots sit closer than that, and at each spot otherwise. On the section above, smoothing at each cell gives nearly the same modules (adjusted Rand index 0.95) for 2.5 times the memory.
- **Minimum cells under the Gaussian.** 10 within three widths, the low end of the authors' 10 to 100. Below it the package sets a location to zero, and zeros raise every correlation, so a low value keeps them few: 0.15% of the locations on the section above. The summary warns when they reach 1%.
- **Infomap.** 20 trials, seed 0.

**Coordinates.** The width is in µm. A dataset in µm or mm is used as it stands, and one in pixels through the pixel size of an image layer that covers its cells. Otherwise the size of the coordinates is unknown: the median spacing between cells is then taken as 9 µm, the median over the cell-segmented datasets this was measured on (7.3 to 12.9 µm), and the summary says so.

**Sections and selections.** Each section is smoothed on its own, as the datasets of the authors' multi-dataset pipeline are, so no section lends counts to another; the correlations are then taken over all of them. The figure shows the sections as the map lays them out, or side by side when they share coordinates, and the summary then gives their order. A section that is too sparse for the width is left out, and the summary names it. Isolate a section, or select cells, to run on those alone.

**Sections of different gene panels.** A section is smoothed on the genes it measured, as a dataset is in the authors' multi-dataset pipeline. The package then counts the others as zero in that section, in the correlations, the lower bound and the map, so genes missing from the same section correlate for that alone, and a module can be made of nothing else. The summary says when this happens and how many modules are mostly such genes. Isolate a section to see its own modules.

## Limitations

**A module is a community in a correlation network, not a test.** No gene or module carries a p-value, and the numbers of modules and genes follow from the three parameters, which have no single right setting. The package's `select_clustering_params` helps choose them, but this tool does not run it.

**Sharing a place is not sharing a cell, and the reverse.** Genes of two cell types that live side by side correlate, and so do genes of one cell type wherever its cells lie. On the section above, shuffling the positions in 20 µm bins, as the lower bound does, removes 95% of the links (107,191 to 5,704), yet 790 genes still form 40 modules, against 2,428 in 47. Four in five of those genes are in the most connected third of their module, and in 11 of the 47 modules, B and plasma cells among them, they are at least half the genes. Such a module says that these genes share cells, which is not yet a pattern of the tissue.

**Cost.** The section above takes about 5 minutes and 7 GB on a server of 124 cores, with up to 10 processes smoothing, twice: once for the modules and once for the lower bound. Each panel of the figure takes about 2.5 s of that. A panel draws at most 50,000 locations, which changes no module. Past 100 modules the panels are drawn smaller, so the figure takes no longer, and past 900 it maps the 900 largest. While it runs, the package writes its panels to a temporary folder, removed at the end of the run.

Smoothie's 2D smoothing means a 3D dataset is not offered the tool.

## Citation

Holdener C, De Vlaminck I. Smoothie: efficient inference and integration of spatial co-expression networks from denoised spatial transcriptomics data. *Communications Biology* 9, 459 (2026). [doi:10.1038/s42003-026-09898-z](https://doi.org/10.1038/s42003-026-09898-z)

The `smoothie-st` package is by Chase Holdener, under the MIT licence.
