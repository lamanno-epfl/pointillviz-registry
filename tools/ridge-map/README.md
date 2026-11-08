# Ridge Map

Answers the question: **How does a gene's expression, or the density of cells, rise and fall across the section, and over which cell types?**

The tool cuts the section into horizontal strips and draws each one as a ridge, like a joy plot laid over the tissue. Ridges are drawn from the back of the figure to the front, and each hides what stands behind it, so the tissue's architecture reads as a relief.

Heights and colors are measured with a Gaussian of width *σ* around each point of a strip, so the strips are cross-sections of one smooth surface. A cell *j* at distance *dⱼ* has the weight *wⱼ = exp(−dⱼ²/2σ²)*. With a gene selected, the height is its local mean expression, where *aⱼ* is the cell's expression in log1p(CP10K):

$$
h(x, y) = \frac{\sum_{j} w_{j} a_{j}}{\sum_{j} w_{j}}
$$

Without a gene, the height is the local cell density:

$$
h(x, y) = \frac{\sum_{j} w_{j}}{2πσ^{2}}
$$

**Figure.** A ridge rises up the figure from its own strip, so a peak belongs to the place where its ridge starts, not to the place where its top appears. The bar in the key shows a round value at its true height; each run has its own scale, so compare runs by these bars. Each stretch of a ridge takes the map's color of the cell type with the most cells around it. The color therefore shows which cell type is most common at that place, not which cells express the gene, and a rare cell type may never be drawn. Ridges drop to their baseline where the tissue ends, so gaps and tears stay black. With several sections, each gets its own figure, all at one scale.

**Summary.** It reports what was drawn, the scale of the heights and the cell types under the tallest ridges, and counts the cells that were not drawn.

The figure is meant for reading the shape of a tissue, not for testing it: heights carry no significance, and tall ridges can hide lower ones behind them.
