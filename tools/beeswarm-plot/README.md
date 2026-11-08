# Beeswarm Plot

Answers the question: **Which genes set each cell type or niche apart, and how is their expression spread across its cells?**

The plot has one row per group of an annotation, such as cell types or niches, and one column per gene. Each glyph is the swarm of the group's cells along the gene's expression, log1p(CP10K), drawn as a smooth outline.

**Plot.** Within a glyph, expression increases from left to right, from the threshold to the 99.5th percentile of the cells above it, on a separate scale for each gene. The glyph's height at each level follows how many of the group's cells have that expression. Its size follows the square root of the share of the group's cells above the threshold, as in the Marker Dot Plot, so a gene that few cells of a group detect draws a small glyph. Its color shows the group's mean expression as a z-score across the groups, from −2.5 to 2.5:

$$
z(g,j) = \frac{x̄(g,j) − mean_{g′} x̄(g′,j)}{sd_{g′} x̄(g′,j)}
$$

Here, *x̄(g,j)* is the mean expression of gene *j* over all cells of group *g*, and the mean and standard deviation (sd) are taken across the groups. Cells at or below the threshold are left out of the glyphs, but the colors always count every cell, so the threshold changes what is drawn, never what a color means. A second, coarse annotation adds an outer color strip and orders the rows: each group is placed with the coarse group that most of its cells belong to.

**Table.** Each row represents one group and one gene. `Cells` gives the group's size, `Mean log1p(CP10K)` its mean expression, `z-score` the value that sets the color, and `% above` the share of its cells above the threshold.
