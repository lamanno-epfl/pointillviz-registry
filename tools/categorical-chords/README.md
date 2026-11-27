# Categorical Chords

Answers the question: **Which categorical groups occur together more or less often than expected?**

The tool compares categorical annotations such as cell types, clusters, or spatial niches. It offers two analysis modes:

- `Spatial contacts` counts pairs of cells from the same section that lie within the selected radius.
- `Annotation relationships` compares two categorical annotations assigned to the same cells, without using spatial coordinates.

For each pair of groups *A* and *B*, the tool compares the observed count *O(A,B)* with the expected count *E(A,B)*:

$$
score(A,B) = log₂(\frac{O(A,B)+1}{E(A,B)+1})
$$

The expectation is calculated from the group frequencies within each section. In annotation mode, *E(A,B) = n(A)n(B)/N*. In spatial-contact mode, it represents the number of *A*-to-*B* pairs expected if the group labels were randomly redistributed over the observed cell positions. A score of *0* indicates agreement with the expectation; positive values indicate enrichment and negative values depletion.

**Chord.** Each sector represents one group. Ribbon width shows the observed number of cells or spatial pairs connecting two groups, while ribbon color shows their *log₂(O/E)* score: orange indicates enrichment and blue depletion. The display limits affect only the chord diagram, which shows the most abundant groups and best-supported relationships.

**Matrix.** Both axes list the selected groups, and each square shows the corresponding *log₂(O/E)* score. The matrix retains all group combinations, including those omitted from the chord diagram.

**Table.** Each row represents one group pair. `Count` gives the observed number of cells or spatial pairs, `Expected` gives the expected count, and `log2 O/E` shows the enrichment or depletion.
