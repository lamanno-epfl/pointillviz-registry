# Smoothie

Answers the question: **Which genes share a spatial pattern in this tissue, and where does each group of genes lie?**

The tool runs [Smoothie](https://caholdener01.github.io/Smoothie/) with the authors' own Python package, `smoothie-st`, and the settings of their pipeline for single-cell data.

First, each gene's normalized expression is smoothed over the tissue. At each location *u*, nearby cells *j* contribute with the Gaussian weights *w(u,j) = exp(−d(u,j)²/2σ²)*, where *d(u,j)* is their distance and *σ* the smoothing width:

$$
s(u) = \frac{\sum_{j} w(u,j) x(j)}{\sum_{j} w(u,j)}
$$

The smoothed maps of every pair of genes *g* and *h* are then compared using the Pearson correlation *r(g,h)*. Two genes are linked when *r(g,h)* exceeds the cutoff *c*, and the clustering power *p* discounts weak links:

$$
a(g,h) = \left(\frac{r(g,h) − c}{1 − c}\right)^{p}
$$

The Infomap algorithm clusters this network into modules, numbered from the largest. Genes without a link join no module.

**Figure.** One panel per module. Each location's module score is the sum of the module's smoothed genes, each scaled from 0 to its own highest value. Grey indicates low scores, red high scores and black the highest within each panel, so compare the patterns of different panels rather than their strengths.

**Table.** Each row represents one module. `Genes` gives its size, and `Members` lists its genes, the most connected first. `Mean margin` averages the margins of its genes, which compare a gene's mean link weight *A* within its module with its highest mean link weight *B* to another module:

$$
margin = \frac{A − B}{max(A, B)}
$$

A margin of **1** means that the gene is linked only within its module, **0** that it is equally close to another module, and negative values that it is closer to another module. A download lists every gene in a module with its network measures.

**Summary.** The package repeats the smoothing and correlations with the cell positions shuffled. The 99.9th percentile of these shuffled correlations is the authors' lowest sensible cutoff, because weaker correlations also arise without any spatial pattern. The summary reports this bound and warns when the cutoff lies below it.

A module groups correlated genes and carries no p-value. Genes of one cell type correlate wherever its cells lie, so a module can reflect shared cells rather than a pattern of the tissue.
