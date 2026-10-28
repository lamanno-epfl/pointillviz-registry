# Cell-Type Percolation

Answers the question: **Which cell types form one connected structure, and which are split into separate patches?**

For each group in the selected annotation, two cells are connected when they are no more than a distance *r* apart. Connections can form chains: if cell A is connected to B and B to C, all three belong to the same patch. As *r* increases, smaller patches merge into larger structures.

**Map.** Each cell is colored by the `log10` size of its connected same-group patch at a fixed linking distance. Brighter colors indicate larger patches, darker colors indicate smaller patches, and unpainted cells have no result.

**Plot.** The x-axis shows the linking distance *r*, normalized by the typical nearest-neighbor distance within each group. The y-axis shows the fraction of the group’s cells contained in its largest connected patch.

$$
P_c(r) = \frac{cells in the largest patch}{all cells in group c}
$$

Each line represents one group. Curves rising further to the left indicate greater connectivity, while curves rising further to the right indicate greater fragmentation.

**Table.** The percolation threshold `r*` is the smallest distance at which the largest patch contains the selected fraction of the group, **50%** by default. `r*/NN` expresses this threshold relative to the group’s own spacing. `r*/null` compares it with random groups of the same size: values below **1** indicate greater connectivity than expected, while values above **1** indicate greater fragmentation.
