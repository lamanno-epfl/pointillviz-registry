# Vector Field

Answers the question: **In which direction does a selected quantity increase across the tissue, and how steeply?**

The tool converts the selected quantity into a smoothed spatial field. Its gradient reveals the direction and strength of local changes across the tissue. Each cell *j* with a valid measurement receives a value *aⱼ*. This is either the selected gene expression, a continuous measurement, or—for a selected population—*1* for cells belonging to that population and *0* for other labelled cells.

At regularly spaced positions *x* across the tissue, the tool calculates a Gaussian-weighted local mean:

$$
A(x) = \sum_{j}w_{j}(x)a_{j},   C(x) = \sum_{j}w_{j}(x),   F(x) = \frac{A(x)}{C(x)}
$$

The weight *wⱼ(x)* decreases with the distance between cell *j* and position *x*, so nearby cells contribute more strongly to *F(x)* than distant cells. For a selected population, *F(x)* represents its smoothed local proportion.

From the spatial variation of this field, the tool calculates its gradient and magnitude:

$$
∇F(x) = \paren{\frac{∂F}{∂x}, \frac{∂F}{∂y}},   g(x) = ‖∇F(x)‖ = √((\frac{∂F}{∂x})² + (\frac{∂F}{∂y})²)
$$

**Map.** Cells are colored by the smoothed local value *F(x)*: dark colors indicate lower values and yellow indicates higher values. The arrows show the gradient *∇F(x)*: their direction indicates the steepest increase, while their length represents the strength *g(x)* of the spatial change. For display, this strength is scaled nonlinearly as *t(x) = √(min(g(x)/g₉₅, 1))*, where *g₉₅* is the 95th percentile of the displayed gradient magnitudes. Arrow color also represents *F(x)*, not gradient magnitude.

**Plot.** To distinguish strong gradients from uncertain estimates, the tool calculates the effective cell count *N(eff)*, the local variance *τ̂²*, and the standard error:

$$
N_{eff}(x) = \frac{(\sum_{j}w_{j}(x))²}{\sum_{j}w_{j}(x)²},   τ̂²(x) = \frac{\sum_{j}w_{j}(x)(a′_{j})²}{C(x)} − (F(x) − ā)²,   SE(x) = \frac{τ̂(x)}{σₚ√(2N_{eff}(x))}
$$

Here, *a′ⱼ = aⱼ − ā* is the centered cell value, and *σₚ* is the spatial width of the Gaussian filter. A small effective cell count or high local variation increases the standard error. The x-axis shows a threshold *t*, while the y-axis shows the proportion of the *M* measurable positions whose standardized gradient magnitude *z(x)* reaches that threshold:

$$
z(x) = \frac{g(x)}{SE(x)},   S_{observed}(t) = \frac{1}{M}\sum_{x∈M}1[z(x) ≥ t],   S_{random}(t) = e^{−t²/2}
$$

The solid curve shows the observed proportion. The dashed curve shows the frequency of standardized two-dimensional gradients expected from the Rayleigh distribution under random cell values. If the observed curve lies clearly above it, strong spatial gradients occur more often than expected by chance. The line at *z = 3* marks the default threshold below which arrows are hidden when filtering is enabled.
