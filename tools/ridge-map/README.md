# Ridge Map

Answers the question: **How does a gene's expression, or the density of cells, rise and fall across the section, and over which cell types?**

The tool cuts the section into horizontal strips and draws each one as a ridge, a joy plot laid over the tissue. Ridges are drawn from the back of the figure to its front, and each hides what stands behind it, so the tissue's architecture reads as a relief.

Every height and colour is measured around its place on the strip with a Gaussian of width *σ*, so the strips are cross-sections of one smooth surface. A cell *j* at distance *dⱼ* weighs *wⱼ = exp(−dⱼ²/2σ²)*. With *aⱼ* its expression in log1p(CP10K), the height is the gene's local mean:

$$
h(x, y) = \frac{\sum_{j} w_{j} a_{j}}{\sum_{j} w_{j}}
$$

With no gene chosen, it is the local cell density:

$$
h(x, y) = \frac{\sum_{j} w_{j}}{2πσ^{2}}
$$

**Height.** The bar in the key shows a round value at its true height. A gene's height is on a log scale, and two runs do not share a scale, so compare them by their bars.

**Place.** A peak rises up the figure from its own strip, so a place on the tissue is where its ridge starts, not the top of the peak.

**Colour.** Each stretch of a ridge takes the map's colour of the cell type with the most cells around it. It says which cell type is most common at that place, not which cells express the gene, so a rare type can be present and never be drawn.

**Edges.** A ridge comes down to its baseline where the tissue stops, so gaps and tears stay black. Cells far outside the section and small stray patches are not drawn, and the summary counts them.

**Sections and selections.** Each section draws its own figure, up to the 8 largest, at one scale. Isolate a section to draw only that one. With cells selected, only they are drawn. A cell whose section did not measure the gene is left out, not counted as zero.

**Outputs.** The figure, the same figure as an SVG download, and a summary of what was drawn and what was not.

The figure is for reading the shape of a tissue, not for testing it: no height carries a significance, and tall ridges can hide a low signal behind them. The tool needs 2D coordinates and a cell-type annotation.
