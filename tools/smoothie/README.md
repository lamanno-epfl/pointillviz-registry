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

On a Xenium 5K section of the mouse maternal–fetal interface (235,803 cells, 5,106 genes), the defaults give 47 modules holding 2,428 genes, among them smooth muscle (Actg2, Myh11), NK cells (Prf1, Gzmb) and the interferon response (Isg15, Ifit1).

**Table.** One row per module: its size, its mean margin and its most connected genes. A gene's margin runs from 1, linked only inside its module, to 0, split evenly between two, so a low mean marks a module that is not sharply set apart.

**Figure.** One panel per module, drawn by the package. Each location's score sums the module's smoothed genes, each scaled by its own highest value, so compare patterns between panels, not their strengths.

**Download.** Every gene in a module, with the package's network measures of it.

**The lower bound.** The package repeats the smoothing and the correlations with the positions shuffled, and the authors use the result as the lowest sensible cutoff. The summary gives it, and says in bold when the cutoff is below it.

**Several sections.** Each section is smoothed on its own, and the correlations are taken over all of them. Genes missing from the same section correlate for that alone, and the summary says when this happens.

Everything not on the form is what the authors' pipeline notebooks give for cell-sized data. A module is a community in a correlation network, not a test: no gene or module carries a p-value. It can also mean that its genes share cells rather than a pattern of the tissue, because genes of one cell type correlate wherever its cells lie.

**Cost.** The section above takes about 5 minutes and 7 GB. The tool needs 2D coordinates.

The `smoothie-st` package is by Chase Holdener, under the MIT licence.
