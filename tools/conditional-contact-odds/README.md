# Conditional Contact Odds

Answers the question: **Which group pairs remain associated after the complete composition of each cell’s neighbourhood is taken into account?**

The tool is based on a softmax model that relates each cell’s group identity to the composition of its local neighbourhood. The probability that cell *i* belongs to group *A* is modeled as:

$$
P(z(i) = A | N(i)) = softmax(h(A) + Σ J(A,B) × N(i,B))
$$

Neighbourhoods are defined by an undirected *k*-nearest-neighbour graph, and *N(i,B)* is the number of group-*B* neighbours connected to cell *i*. The model jointly estimates the group-specific offsets *h(A)* and the coefficients *J(A,B)* from the observed group labels. While *h(A)* accounts for the baseline tendency of a cell to belong to group *A*, the main quantities of interest are the coefficients *J(A,B)*. They describe the association between groups *A* and *B* after the complete neighbourhood composition has been taken into account.

More precisely, while all other neighbour counts remain fixed, one additional *B*-neighbour changes the log-odds of group *A* versus group *C* by *J(A,B) − J(C,B)*.

**Map.** Using the fitted softmax model above, each cell is colored by *log P(z(i) = observed group | N(i))*, the log probability assigned to its observed group, given its neighbourhood. Values closer to **0** indicate that the model assigns a high probability to the observed group, while more negative values indicate that the cell’s identity is unusual for its neighbourhood.

**Matrix.** Both axes list all groups, and each square shows the corresponding fitted coefficient *J(A,B)*. The coefficients are centered to have a mean of zero, *J′(A,B) = J(A,B) − mean(J)*. Positive values indicate stronger-than-average conditional association; negative values indicate weaker association or relative avoidance. The diagonal shows conditional self-association.

**Table.** Each row represents one group pair. `Conditional log-odds` is the centered *J(A,B)* value from the matrix. `Marginal log2 obs/exp` is the Cell-Type Co-occurrence score calculated on the same neighbour graph, without accounting for other groups. `Contacts` is the number of observed connections between the groups. `Rank shift` shows how the pair’s position changes when pairs are ranked by conditional association instead of ordinary co-occurrence. A positive value means that the pair ranks higher by conditional association, while a negative value means that it ranks lower.
