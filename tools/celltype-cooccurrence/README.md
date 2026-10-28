# Cell-Type Co-occurrence

Answers the question: **Which cell types are found near each other more often
than expected by chance?**

Both axes list all groups in the selected annotation, so each square represents
one group pair. Its color shows the co-occurrence score within the selected
radius: orange to light yellow indicates more neighboring pairs than expected
under random mixing, while blue indicates fewer pairs than expected. The matrix
is symmetrical, and its diagonal shows whether cells from the same group
cluster together.

More specifically, the score is the log2-transformed ratio between the observed
and expected numbers of neighboring cell pairs:

$$
score(i,j) = log₂(\frac{observed(i,j) + 1}{expected(i,j) + 1})
$$

The observed value is the number of pairs from groups *i* and *j* found within
the selected radius. The expected value represents random mixing and is
calculated as:

$$
expected(i,j) = 2 × Nₚₐᵢᵣₛ × pᵢ × pⱼ
$$

Here, *Nₚₐᵢᵣₛ* is the total number of neighboring pairs, while *pᵢ* and
*pⱼ* are the overall proportions of the two groups. Thus, a score of **0**
means that there are as many observed pairs as expected, while scores of **+1**
and **+2** indicate approximately **2×** and **4×** the expected number,
respectively.

## Sweeping the radius

The radius can be **swept**: the tool's panel offers *Sweep from … to … in …
steps* under the Radius control. **Run sweep** runs the tool once per radius,
all in one process, and a slider then moves between the results — the matrix
and the table at each radius, every one the tool's own answer at that radius,
with nothing re-run while you drag. Seeing every radius, not only the one that
looked best, is the point: a pair that is enriched at one radius and not at its
neighbours is a pair to distrust.
