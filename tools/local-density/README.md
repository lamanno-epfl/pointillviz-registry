# Local Cell Density

Answers the question: **How densely are cells packed around each individual cell?**

For each cell *i*, the tool measures the distance *rᵢ* to its *k*-th nearest neighbour within the same section. This distance defines a circle in 2D or a sphere in 3D around the cell that contains its *k* nearest neighbours. A smaller circle or sphere indicates more closely packed cells and therefore a higher local density. Local density is calculated as:

$$
densityᵢ = \frac{k}{πrᵢ²}
$$

in 2D, and:

$$
densityᵢ = \frac{k}{(4/3)πrᵢ³}
$$

in 3D.

**Map.** Each cell is colored by its local density: brighter colors indicate higher density, darker colors lower density, and unpainted cells have no result.

**Plot.** Cells are sorted by local density. The x-axis shows the percentage of cells, and the y-axis shows the corresponding density. A point at (80%, y) means that 80% of cells have a local density of y or lower, while 20% have a higher density. At 50%, the curve therefore shows the median density.
