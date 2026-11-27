# Expression Hotspots (Getis-Ord Gi*)

Answers the question: **Where does a selected gene form spatial clusters of high or low expression?**

For a selected gene, the tool considers each cell together with its *k* nearest neighbours and calculates a Getis-Ord `Gi*` z-score. This score measures whether gene expression in that local neighbourhood is higher or lower than expected if the observed expression values were randomly distributed across the cells.

More specifically, the `Gi*` score expresses the difference between the observed and expected neighbourhood expression in units of the variation expected under random mixing:

$$
Gᵢ* = \frac{observed local expression − expected local expression}{expected random variation} = \frac{\sum_{j∈Nᵢ}xⱼ − mᵢx̄}{S√(mᵢ(n−mᵢ)/(n−1))}
$$

Here, *Nᵢ* contains cell *i* and its *k* nearest neighbours, *mᵢ* is its size, *xⱼ* is the selected gene’s expression in cell *j*, and *x̄* and *S* are its mean and standard deviation across all *n* mapped cells.

**Map.** Each cell is colored by its `Gi*` z-score. Red shows locally elevated expression, while blue shows locally reduced expression. Because `Gi*` includes the cell itself, the map describes the expression level of the whole local region rather than whether one cell differs from its neighbours.

**Summary.** `Gi*` z-scores are converted into two-sided p-values and corrected across cells using the Benjamini–Hochberg method by default. Click the **info icon** after the run to see how many cells were classified as significant hotspots or coldspots at the selected q-value threshold.

**Table.** Each row represents one cell type. `In hotspot` and `In coldspot` count its cells in significant regions. `Hotspot fraction` gives the proportion of that cell type found in hotspots, while `Mean Gi*` shows its average hotspot score.
