# Heavy-Traffic Convergence Simulation

Companion code for the figure illustrating the heavy-traffic limit
$\epsilon q \to \mathrm{Expo}\!\left(2/(\sigma_a^2 + \sigma_s^2)\right)$
in the tutorial:

> Sushil Mahavir Varma, Prakirt Jhunjhunwala, Daniela Hurtado-Lange, and
> Siva Theja Maguluri. *Transform Method for Stochastic Processing and
> Matching Networks.* INFORMS TutORials in Operations Research, 2026.

The script simulates a discrete-time G/G/1 queue at four values of the
heavy-traffic parameter $\epsilon$ and compares the empirical
distribution of the scaled queue length $\epsilon q$ against the
exponential limit predicted by heavy-traffic theory.

## What it produces

Running the script generates six figure files next to the script:

| File | Contents |
| --- | --- |
| `ht_convergence.pdf`, `ht_convergence.png` | Combined 2×2 figure (densities + tail CCDFs for both cases). |
| `ht_bernoulli_density.eps` | Bernoulli arrivals + Bernoulli service, density panel. |
| `ht_bernoulli_tail.eps` | Bernoulli case, tail CCDF panel. |
| `ht_bursty_density.eps` | Bursty (K=4) arrivals + Bernoulli service, density panel. |
| `ht_bursty_tail.eps` | Bursty case, tail CCDF panel. |

## What is simulated

For each $\epsilon \in \{0.2, 0.1, 0.05, 0.02\}$ and each of the two
arrival/service cases below, the script simulates the Lindley recursion
```
q(k+1) = max(q(k) + a(k) - s(k), 0)
```
for $T$ slots, discards the first half as burn-in, and records the
empirical distribution of $\epsilon q$ from the remainder. Vectorization
is done via the reflection identity
$q(k) = S(k) - \min_{0 \le j \le k} S(j)$, where
$S(k) = \sum_{i<k} (a(i) - s(i))$.

| Case | Arrivals | Service | $\sigma_a^2 + \sigma_s^2$ | Limit |
| --- | --- | --- | --- | --- |
| 1 | Bernoulli($\lambda$) | Bernoulli($\mu$) | $\to 0.5$ | $\mathrm{Expo}(4)$ |
| 2 | $4 \cdot$ Bernoulli($\lambda/4$) | Bernoulli($\mu$) | $\to 2.0$ | $\mathrm{Expo}(1)$ |

with $\lambda = 0.5 - \epsilon/2$ and $\mu = 0.5 + \epsilon/2$.

Per-epsilon simulation lengths:

| $\epsilon$ | $T$ |
| --- | --- |
| 0.20 | 128,000,000 |
| 0.10 | 128,000,000 |
| 0.05 | 128,000,000 |
| 0.02 | 256,000,000 |

## Installation

Requires Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

## Running

```bash
python3 ht_convergence_sim.py
```

Wall-clock time: about 30 seconds on a modern laptop. Peak memory: about
2 GB (at $T = 256\,\mathrm{M}$). Output files are written to the
directory containing the script.

## Reproducibility

The simulation uses `numpy.random.default_rng(SEED)` with a fixed seed
(`SEED = 20260527`), so the curves are deterministic on a given numpy
version. The figures in the manuscript were generated with:

- Python 3.10
- numpy 2.2.6
- matplotlib 3.10.8

If you need byte-identical EPS output, pin these versions in
`requirements.txt`. Otherwise the figures will look the same but EPS
byte streams may differ across matplotlib releases.

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code or adapt it, please cite the tutorial:

```bibtex
@incollection{transform_method_tutorial_2026,
  title     = {Transform Method for Stochastic Processing and Matching Networks},
  author    = {Varma, Sushil Mahavir and Jhunjhunwala, Prakirt and
               Hurtado-Lange, Daniela and Maguluri, Siva Theja},
  year      = {2026},
  booktitle = {INFORMS TutORials in Operations Research},
  publisher = {INFORMS},
}
```
