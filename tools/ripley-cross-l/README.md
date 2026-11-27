# Ripley’s Cross-L Function

Answers the question: **At what distances does group B accumulate around or avoid group A?**

The tool uses group *A* as the reference and measures the distances from its cells to cells from group *B*. It gradually increases a radius *r* around every *A*-cell and counts how many *B*-cells fall within that radius.

More specifically, let *C(A,B)(r)* be the number of observed *A*-to-*B* pairs separated by at most *r*. The cross-K function adjusts this count for the overall density *ρ(B)* of group *B* and for *m(A)(r)*, the number of *A*-cells included at radius *r*:

$$
K(A,B)(r) = \frac{C(A,B,observed)(r)}{ρ(B)m(A)(r)} vs. \frac{C(A,B,expected)(r)}{ρ(B)m(A)(r)} = πr²
$$

Under complete spatial randomness, *K(A,B)(r) = πr²*. The tool transforms this comparison into a curve centered around zero:

$$
L(A,B)(r)−r = √(\frac{K(A,B)(r)}{π})−r
$$

Positive values indicate more *B*-cells around *A* than expected, while negative values indicate fewer.

**Plot.** The x-axis shows the radius *r*, and the y-axis shows *L(A,B)(r) − r*. The solid line is the observed curve; its peak indicates the spatial scale at which *B* is most strongly accumulated around *A*. The grey band comes from random redistributions of the *B* labels while keeping *A* fixed. Values above the band indicate enrichment of *B* around *A*, while values below indicate spatial avoidance.

**Table.** Each row represents one tested radius *r*. `L(r) − r` is the observed curve value, while `Envelope low` and `Envelope high` are the lowest and highest values from the random relabellings at that radius. `Outside envelope` indicates whether the observed value lies above or below this range; a blank entry means that it lies inside.
