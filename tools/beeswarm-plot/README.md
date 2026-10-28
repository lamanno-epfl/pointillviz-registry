# Beeswarm Plot

Answers the question: **Which genes set each cell type or niche apart, and how are they spread across its cells?**

The plot has one row per group of an annotation (cell types, niches or any other categorical column) and one column per gene. Each glyph is the swarm of the group's cells along the gene's expression, log1p(CP10K):

- **Length.** Low expression is at the glyph's left end and high expression at its right. Each gene has its own scale, from the threshold to the 99.5th percentile of the cells above it.
- **Height.** At each level, the height follows how many of the group's cells are there.
- **Size.** The glyph's size follows the share of the group's cells above the threshold (its square root, as the Marker Dot Plot sizes its dots). A gene that few cells of a group detect draws a small glyph.
- **Colour.** The group's mean expression as a z-score across the groups, in one of the viewer's gene colormaps, from −2.5 to 2.5:

$$
z_{g,j} = \frac{\bar{x}_{g,j} - \operatorname{mean}_{g'}\,\bar{x}_{g',j}}{\operatorname{sd}_{g'}\,\bar{x}_{g',j}}
$$

where $\bar{x}_{g,j}$ is the mean log1p(CP10K) of gene $j$ over every cell of group $g$.

**Genes.** The form opens on the 20 genes detected in the most cells, the Marker Dot Plot's Top markers. Search to add genes (up to 40), or remove them.

**Groups.** A row needs at least 10 cells, and the plot draws at most the 120 largest groups. A second, coarse annotation adds an outer colour strip and orders the rows: each group takes the coarse group most of its cells belong to.

**Threshold.** Cells at or below the threshold are left out of the swarms; 0 keeps every cell that detects the gene. The colours always count every cell of a group, so the threshold changes what is drawn, never what a colour means.

**Outputs.** The plot as a figure and as an SVG download, and a table with each group's cells, mean expression, z-score and share above the threshold for every gene.
