# Neighborhood Composition & Mixing

Answers the question: **How mixed is each cell’s neighbourhood, and which groups occur next to one another more or less often than expected?**

For each cell *i*, the tool finds its *k* nearest neighbours and records how many belong to each group.

**Map.** Each cell is colored by the normalized Shannon entropy of its neighbourhood:

$$
Mᵢ = \frac{−\sum_{j}pᵢⱼlog(pᵢⱼ)}{log(T)}
$$

Here, *pᵢⱼ* is the proportion of cell *i*’s labelled neighbours belonging to group *j*, and *T* is the number of groups present in the dataset. A score of **0** means that all labelled neighbours belong to the same group. Higher values indicate that several groups are represented more evenly.

**Matrix.** Both axes list all groups. Each square compares the observed number *O(A,B)* of *A*-to-*B* neighbour relations with the expected number *E(A,B)*:

$$
score(A,B) = log₂(\frac{O(A,B)+1}{E(A,B)+1})
$$

A score of **0** means that the observed and expected numbers match. Scores of **+1** and **−1** indicate approximately twice and half the expected number, respectively.

The expected number is calculated by multiplying the total number of valid neighbour positions around all *A*-cells (usually about *n(A)* × *k*) by the proportion of valid neighbour positions across all groups occupied by *B*-cells. Because the calculation counts neighbour positions rather than unique cells, the same cell may be counted multiple times, depending on its position and how many other cells include it among their *k* nearest neighbours.

**Table.** Each row summarizes one group. `Cells scored` is the number of its cells with a valid mixing score, while `Mean mixing` is their average *Mᵢ*. `Self log2 O/E` is the group’s diagonal matrix value and indicates whether its cells cluster together or are more dispersed. `Most enriched neighbour` is the other group with the highest matrix score, and the accompanying `log2 O/E` column gives that score.
