# Ridge Map

Answers the question: **How does a gene's expression, or the density of cells, rise and fall across the section, and over which cell types?**

The tool cuts the section into horizontal strips and draws each one as a ridge, a joy plot laid over the tissue. Ridges are drawn from the back of the figure to its front, and each hides what stands behind it, so the tissue's architecture reads as a relief.

Every height and colour is measured around its place on the strip with a Gaussian of width *σ*, so the strips are cross-sections of one smooth surface and a peak carries on from one strip to the next. A cell *j* at distance *dⱼ* from that place weighs *wⱼ = exp(−dⱼ²/2σ²)*, by its true position, so a regular lattice of pixels draws flat. With *aⱼ* its expression in log1p(CP10K), the height is the gene's local mean:

$$
h(x, y) = \frac{\sum_{j} w_{j} a_{j}}{\sum_{j} w_{j}}
$$

With no gene chosen, it is the local cell density:

$$
h(x, y) = \frac{\sum_{j} w_{j}}{2πσ^{2}}
$$

**Figure.** The section keeps its proportions and the orientation of the map.

- **Height.** The tallest ridge stands *Relief* strip heights tall. The bar in the key shows a round value at its true height, and it applies to every ridge. A gene's height is on a log scale: a ridge twice as tall is far more than twice the expression. The figures of one run share one scale, but two runs do not: compare them by their bars.
- **Place.** A peak stands behind its baseline: it rises up the figure from its own strip and overlaps the strips further back. A place on the tissue is where its ridge starts, not the top of the peak.
- **Colour.** Each stretch of a ridge takes the map's colour of the cell type with the most cells around it, counted with the same weights. It says which cell type is most common at that place, not which cells express the gene. A colour with less than 3:1 contrast on black is lightened just enough, keeping its hue, and the summary says so. A stretch shorter than 5 px takes the colour of its longer neighbour, at any number of strips. The soft glow under a ridge is the same colour, brighter the higher the ridge stands within its stretch of tissue, and adds no measurement.
- **Edges.** A ridge is drawn where the Gaussian holds at least 10% of what it holds around the section's typical cell. Along the strip, it comes down to its baseline over 3*σ* at each end, where the tissue stops, so gaps and tears stay black, and the heights in that margin are lower than measured. Patches holding fewer than 0.05% of the section's cells are taken for strays and not drawn. So are cells far outside the section, past an empty gap as wide as the middle 90% of the section, and they do not change the figure's scale. A distant piece holding more than 5% of the cells counts as tissue.

**Cells that say nothing.** A cell with no coordinates cannot be drawn. A cell whose section did not measure the gene is left out as well: it does not count as zero, and a section that did not measure the gene draws no figure. Cells with no cell type count toward the height but not the colours; a stretch with no labelled cell around it is grey, named Unlabelled in the key. The summary gives their number when a grey stretch is drawn, or when they reach 1% of the cells drawn.

**Sections and selections.** Each section draws its own figure, up to the 8 largest, at one scale of length, strip height and ridge height, and names itself in its key. Isolate a section to draw only that one. With cells selected, only they are drawn, and the figure closes in on them.

**Outputs.** The figure, the same figure as an SVG download, and a summary: the cells and strips drawn, the scale of this run, and the cell types over which the ridges stand tallest on average. It counts the cells not drawn, by reason: no coordinates, gene not measured, far outside the section, in a section left out, sparse or stray. It also counts the cell types present but not drawn, because none is the most common over a long enough stretch, and names them when there are at most five.

## Parameters

- **Gene.** Sets the height of the ridges. With no gene chosen, the height is the cell density.
- **Strips.** How many strips the section is cut into. Fewer are cut when each would hold fewer than 15 cells on average, when they would lie closer than 2.5 px, or when the figures would still not fit the 8 MiB output limit with fewer points on each ridge and longer stretches of one colour. The limit is measured on the figures themselves, and the summary says when this happens. A strip with no tissue in it draws no ridge.
- **Relief.** How tall the tallest ridge stands, in strip heights, up to half the figure's height.
- **Smoothing.** The width *σ* of the Gaussian, in strip heights. It widens by itself when it would hold fewer than 5 cells on average, or when it would be narrower than the figure can draw, and the summary says so.

## Limitations

The figure is for reading the shape of a tissue, not for testing it: no height carries a significance. Tall ridges hide what stands behind them, so a low signal next to a high one can be out of sight; lower the relief, or read the place on the map.

Colour shows only the most common cell type, so a rare type can be present and never be drawn.

The tool needs 2D coordinates and a cell-type annotation. A 3D dataset is not offered it.
