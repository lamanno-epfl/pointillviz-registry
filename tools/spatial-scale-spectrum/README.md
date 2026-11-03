# Spatial Scale Spectrum

Answers the question: **At which spatial scales is each population organized into regular patterns, small patches, or large territories?**

The tool describes the spatial distribution of a population as a combination of waves of different sizes and shows at which wavelengths its spatial patterns are strongest.

To describe these patterns mathematically, the tool applies a Fourier transform. For each population *c* and wavevector *k*, it calculates the following complex coefficient:

$$
f̂(k) = \sum_{j}f(x_{j})e^{−i k · x_{j}}   ⟶   T_{c}(k) = \sum_{j}(z_{j}−p_{c})e^{−i k · x_{j}}
$$

Here, *zⱼ = 1* if cell *j* belongs to population *c*, and *zⱼ = 0* for all other labelled cells. *p(c) = N(c)/N* is the overall proportion of population *c*, and *xⱼ* is the position of cell *j*. This term represents a spatial wave at that position. Its direction and wavelength are determined by *k*, while the available values of *k* depend on the selected binning resolution *G*. For each section, the cells are placed on a square grid of physical side length *L*, giving:

$$
k = \paren{\frac{2π n_{x}}{L}, \frac{2π n_{y}}{L}},   λ(k) = \frac{2π}{|k|}
$$

where *n(y) = −G/2, …, G/2−1* and *n(x) = 0, …, G/2*.

**Map.** An inverse Fourier transform of all coefficients would reconstruct the complete mean-centered population field. Instead, the map reconstructs only the wavelength band *B* containing the automatically determined patch scale, defined as the first wavelength at which the spectrum reaches halfway between *S = 1* and its long-wavelength plateau:

$$
B = \set{k : λ_{low} ≤ \frac{2π}{|k|} < λ_{high}},   g_{c}(x) = F⁻¹[M_{B}(k)T_{c}(k)](x) ∝ \sum_{k in B}T_{c}(k)e^{i k · x}
$$

Each cell receives the reconstructed signal at its position, normalized by the average number of cells per occupied grid bin: *Vᵢ = g(c,xᵢ)/ρ(bin)*. Warm colors indicate positions where the retained waves combine into a positive signal, cool colors indicate a negative signal, and values near zero appear neutral.

**Plot.** The plot shows how strongly population *c* is spatially organized at each wavelength *λ*. The x-axis shows wavelength on a logarithmic scale, while the y-axis shows the normalized spatial power *S(c,λ)*. A value of *S(c,λ) = 1* is expected when the population labels are randomly distributed over the observed cell positions. Values above **1** indicate stronger spatial patterns, such as patches or large territories, while values below **1** indicate a more even arrangement. Each colored line represents one population, and the grey band shows the variation obtained from random label redistributions.

More specifically, the tool normalizes the squared magnitude of each Fourier coefficient and then averages waves with similar wavelengths across directions:

$$
S_{c}(k) = \frac{|T_{c}(k)|²}{p_{c}(1−p_{c})\frac{N}{N−1}(N−S_{all}(k))},   S_{all}(k) = \frac{|\sum_{j}e^{−i k · x_{j}}|²}{N},   S_{c}(λ_{b}) = \frac{1}{M_{b}}\sum_{k in b}S_{c}(k)
$$

Here, *S(all,k)* accounts for the observed cell arrangement, *p(c)* for population abundance, and *M(b)* is the number of Fourier modes in wavelength shell *b*.

**Table.** Each row represents one population. `S short` measures spatial organization across the shortest resolved wavelengths, while `S long` measures it across the longest. `Patch scale` marks the wavelength where *S* lies halfway between **1** and its long-wavelength level and determines the wavelength band used for the map. `Spectral exponent` describes how *S* changes toward longer wavelengths, with negative values indicating increasingly strong spatial organization at larger scales. When the real-space check is enabled, `Var / Var_null` compares the observed variation in population counts across spatial windows with the random-label expectation, while `alpha` describes how quickly this variation grows as the window size increases.
