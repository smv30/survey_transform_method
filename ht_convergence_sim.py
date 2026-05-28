"""
Heavy-traffic convergence of the discrete-time G/G/1 queue.

Companion code for the figure illustrating the heavy-traffic limit
    eps * q  -->  Expo(2 / (sigma_a^2 + sigma_s^2))
in the tutorial "Transform Method for Stochastic Processing and Matching
Networks" (INFORMS TutORials in Operations Research).

For each value of the heavy-traffic parameter epsilon, we simulate the
discrete-time Lindley recursion
    q(k+1) = max(q(k) + a(k) - s(k), 0)
and compare the empirical distribution of eps*q against the limiting
exponential predicted by the heavy-traffic theory. We do this for two
contrasting arrival/service distributions:
  1) Bernoulli arrivals and Bernoulli service (sigma^2_total -> 0.5).
  2) Bursty (K=4) arrivals + Bernoulli service (sigma^2_total -> 2.0).

Per-epsilon simulation length T (see get_T()):
  T = 128,000,000  for eps in {0.2, 0.1, 0.05}
  T = 256,000,000  for eps = 0.02
Half of each simulation is discarded as burn-in; the effective number of
independent samples is roughly N_eff ~ (T/2) * eps^2.

Implementation notes:
  - The Lindley recursion is vectorized via the reflection identity
        q(k) = S(k) - min_{0 <= j <= k} S(j),
    with S(k) = sum_{i<k} (a(i) - s(i)).
  - Random arrivals are drawn in chunks of 8 million to bound the
    int64 -> int32 conversion spike.
  - run_eps() frees intermediate arrays eagerly so peak working memory
    stays around 2 int32 arrays of length T (~2 GB at T = 256M).

Outputs (all written next to this script):
  - ht_convergence.pdf / ht_convergence.png : combined 2x2 figure
    (densities and tail CCDFs for both cases).
  - ht_bernoulli_density.eps : single panel.
  - ht_bernoulli_tail.eps    : single panel.
  - ht_bursty_density.eps    : single panel.
  - ht_bursty_tail.eps       : single panel.

Usage:
  python3 ht_convergence_sim.py

Dependencies:
  - Python 3.9+
  - numpy >= 1.20
  - matplotlib >= 3.4

The random seeds are fixed in SEED below, so the output is fully
reproducible.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
SEED = 20260527

# ---------------------------------------------------------------------------
# Style: color palette + markers, sorted ascending by epsilon.
# ---------------------------------------------------------------------------
EPS_STYLE = {
    0.02: dict(color="#067BC2", marker="D"),   # Bright Teal Blue, diamond
    0.05: dict(color="#84BCDA", marker="^"),   # Sky Reflection,   up triangle
    0.10: dict(color="#F37748", marker="s"),   # Coral Glow,       square
    0.20: dict(color="#D56062", marker="o"),   # Lobster Pink,     circle
}
THEORY_STYLE = dict(color="black", linestyle="--", linewidth=4.0)

FS_TITLE = 19
FS_LABEL = 19
FS_TICK = 17
FS_LEGEND = 17
LW_LINE = 2.6
MS_MARKER = 12
N_MARKERS_TARGET = 6


def fmt_rate(rate):
    """Format the exponential rate for the legend (drop trailing zeros)."""
    if abs(rate - round(rate)) < 1e-9:
        return str(int(round(rate)))
    return f"{rate:g}"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
# Chunked samplers: rng.binomial returns int64 internally, so generating
# T samples in one shot temporarily holds 8T bytes (int64) plus the int32
# target (4T bytes) -- prohibitive at T = 256M. By doing the conversion
# chunk-by-chunk we cap the spike at a few tens of MB.
_SAMPLE_CHUNK = 8_000_000


def bernoulli_sampler(rng, T, p):
    out = np.empty(T, dtype=np.int32)
    for start in range(0, T, _SAMPLE_CHUNK):
        end = min(start + _SAMPLE_CHUNK, T)
        out[start:end] = rng.binomial(1, p, size=end - start)
    return out


def bursty_sampler_factory(K):
    def sampler(rng, T, m):
        p = m / K
        out = np.empty(T, dtype=np.int32)
        for start in range(0, T, _SAMPLE_CHUNK):
            end = min(start + _SAMPLE_CHUNK, T)
            out[start:end] = K * rng.binomial(1, p, size=end - start)
        return out
    return sampler


def run_eps(arrival_sampler, service_sampler, eps, T, base, rng, burn_frac=0.5):
    """Inline Lindley recursion with aggressive freeing.

    Peak working memory is ~2 int32 arrays of length T (e.g., a + S, or
    S + running_min), so ~2 GB at T = 256M.
    """
    lam = base - eps / 2
    mu = base + eps / 2

    a = arrival_sampler(rng, T, lam)               # int32, T
    s = service_sampler(rng, T, mu)                # int32, T

    np.subtract(a, s, out=a)                       # a is now d = a - s
    del s                                          # free 1 array

    S = np.empty(T + 1, dtype=np.int32)            # +1 array
    S[0] = 0
    np.cumsum(a, out=S[1:])
    del a                                          # free 1 array

    running_min = np.minimum.accumulate(S)         # +1 array
    np.subtract(S, running_min, out=S)             # S is now q (in place)
    del running_min                                # free 1 array

    burn = int(burn_frac * (T + 1))
    out = S[burn:].copy()                          # +0.5 array briefly
    del S                                          # free 1 array
    return out


def extract_plot_data(q_samples, eps, T_eps):
    max_q = int(q_samples.max())
    counts = np.bincount(q_samples, minlength=max_q + 1)
    pmf = counts / counts.sum()
    x = np.arange(max_q + 1) * eps
    density = pmf / eps
    cdf = np.cumsum(pmf)
    ccdf = np.clip(1.0 - cdf, 0.0, 1.0)
    emp_mean = float((eps * q_samples).mean())
    N_eff = T_eps * eps ** 2 / 2.0
    return {
        "x": x, "density": density, "ccdf": ccdf,
        "emp_mean": emp_mean, "N_eff": N_eff, "T_eps": T_eps,
    }


def get_T(eps):
    if eps <= 0.03:
        return 256_000_000
    return 128_000_000


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def aggregate(x, y, target_dx):
    """Combine consecutive bins of width dx into wider bins of width >= target_dx
    by averaging. For visual smoothing of multi-modal empirical PMFs."""
    if len(x) < 2:
        return x, y
    dx = float(x[1] - x[0])
    if dx >= target_dx:
        return x, y
    k = int(np.ceil(target_dx / dx))
    n_new = len(x) // k
    if n_new == 0:
        return x, y
    x_agg = x[: n_new * k].reshape(n_new, k).mean(axis=1)
    y_agg = y[: n_new * k].reshape(n_new, k).mean(axis=1)
    return x_agg, y_agg


def marker_step_for(x_arr, x_max, n_markers=N_MARKERS_TARGET):
    """Return a markevery step that yields ~n_markers markers in [0, x_max]."""
    n_in = int(np.sum(x_arr <= x_max))
    if n_in <= n_markers:
        return 1
    return max(1, n_in // n_markers)


def style_axes(ax):
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_density(ax, plot_data, theory_mean, title, eps_list,
                 aggregate_dx=None):
    x_max = 5 * theory_mean
    for eps in sorted(eps_list):
        d = plot_data[eps]
        x, density = d["x"], d["density"]
        if aggregate_dx is not None:
            x, density = aggregate(x, density, aggregate_dx)
        mask = x <= x_max
        x_p, y_p = x[mask], density[mask]
        style = EPS_STYLE[eps]
        step = marker_step_for(x_p, x_max)
        ax.plot(
            x_p, y_p,
            linestyle="-", linewidth=LW_LINE,
            color=style["color"], marker=style["marker"],
            markersize=MS_MARKER, markevery=step,
            markeredgecolor="white", markeredgewidth=0.6,
            label=rf"$\epsilon = {eps}$",
        )
    xs = np.linspace(0, x_max, 400)
    rate = 1.0 / theory_mean
    ax.plot(xs, rate * np.exp(-xs * rate),
            label=rf"Expo({fmt_rate(rate)}) limit", **THEORY_STYLE)
    ax.set_xlabel(r"$\epsilon\,q$", fontsize=FS_LABEL)
    ax.set_ylabel("density", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)
    style_axes(ax)
    ax.legend(loc="best", fontsize=FS_LEGEND, framealpha=0.92)


def plot_ccdf(ax, plot_data, theory_mean, title, eps_list):
    x_max = 6 * theory_mean
    for eps in sorted(eps_list):
        d = plot_data[eps]
        mask = (d["x"] <= x_max) & (d["ccdf"] > 0)
        x_p, y_p = d["x"][mask], d["ccdf"][mask]
        style = EPS_STYLE[eps]
        step = marker_step_for(x_p, x_max)
        # Use 'steps-post' drawstyle so we get a step plot that also supports
        # markers at the data points.
        ax.plot(
            x_p, y_p,
            drawstyle="steps-post", linestyle="-", linewidth=LW_LINE,
            color=style["color"], marker=style["marker"],
            markersize=MS_MARKER, markevery=step,
            markeredgecolor="white", markeredgewidth=0.6,
            label=rf"$\epsilon = {eps}$",
        )
    xs = np.linspace(0, x_max, 400)
    rate = 1.0 / theory_mean
    ax.plot(xs, np.exp(-xs * rate),
            label=rf"Expo({fmt_rate(rate)}) limit", **THEORY_STYLE)
    ax.set_xlabel(r"$\epsilon\,q$", fontsize=FS_LABEL)
    ax.set_ylabel(r"$\mathbb{P}(\epsilon q > x)$", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_yscale("log")
    ax.set_xlim(0, x_max)
    ax.set_ylim(1e-5, 1.1)
    style_axes(ax)
    ax.legend(loc="best", fontsize=FS_LEGEND, framealpha=0.92)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_case(arrival_sampler, service_sampler, eps_list, base, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for eps in eps_list:
        T = get_T(eps)
        print(f"    eps = {eps:6.3f}  T = {T:>12,}")
        samples = run_eps(arrival_sampler, service_sampler, eps, T, base, rng)
        out[eps] = extract_plot_data(samples, eps, T)
        del samples
    return out


def save_single_panel_eps(plot_func, plot_data, theory_mean, title,
                          eps_list, path, **kwargs):
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    plot_func(ax, plot_data, theory_mean, title, eps_list, **kwargs)
    fig.tight_layout()
    fig.savefig(path, format="eps", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    # Print environment info to aid reproducibility debugging if curves
    # later look different on someone else's machine.
    import sys
    import matplotlib
    print(f"Python   : {sys.version.split()[0]}")
    print(f"numpy    : {np.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"Random seed: {SEED}")
    print()

    eps_list = [0.2, 0.1, 0.05, 0.02]
    base = 0.5

    print("Case 1: Bernoulli arrivals + Bernoulli service")
    bern_data = run_case(bernoulli_sampler, bernoulli_sampler,
                         eps_list, base, SEED)
    bern_sigma2 = 2 * base * (1 - base)
    bern_mean = bern_sigma2 / 2

    print("Case 2: Bursty (K=4) arrivals + Bernoulli service")
    K = 4
    burst_data = run_case(bursty_sampler_factory(K), bernoulli_sampler,
                          eps_list, base, SEED + 10)
    burst_sigma2 = base * (K + 1 - 2 * base)
    burst_mean = burst_sigma2 / 2

    print("\nSanity check (empirical vs. theoretical mean of eps*q):")
    print(f"  {'eps':>6}  {'N_eff':>10}  "
          f"{'Bern emp':>10}  {'Bern thy':>10}  "
          f"{'Burst emp':>10}  {'Burst thy':>10}")
    for eps in eps_list:
        b = bern_data[eps]
        u = burst_data[eps]
        print(
            f"  {eps:6.3f}  {b['N_eff']:10,.0f}  "
            f"{b['emp_mean']:10.4f}  {bern_mean:10.4f}  "
            f"{u['emp_mean']:10.4f}  {burst_mean:10.4f}"
        )

    # Titles used in panels
    bern_density_title = (
        r"Bernoulli arrivals + Bernoulli service  "
        r"($\sigma_a^2 + \sigma_s^2 \to 0.5$)"
    )
    burst_density_title = (
        r"Bursty ($K=4$) arrivals + Bernoulli service  "
        r"($\sigma_a^2 + \sigma_s^2 \to 2.0$)"
    )
    bern_tail_title = "Bernoulli: tail"
    burst_tail_title = r"Bursty ($K=4$): tail"

    burst_aggregate_dx = 0.40

    # ---- Combined 2x2 figure (PDF + PNG) ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    plot_density(axes[0, 0], bern_data, bern_mean,
                 bern_density_title, eps_list)
    plot_density(axes[0, 1], burst_data, burst_mean,
                 burst_density_title, eps_list,
                 aggregate_dx=burst_aggregate_dx)
    plot_ccdf(axes[1, 0], bern_data, bern_mean, bern_tail_title, eps_list)
    plot_ccdf(axes[1, 1], burst_data, burst_mean, burst_tail_title, eps_list)
    fig.suptitle(
        r"Heavy-traffic convergence of $\epsilon q$ to "
        r"$\mathrm{Expo}\!\left(2/(\sigma_a^2 + \sigma_s^2)\right)$",
        fontsize=FS_TITLE + 1,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "ht_convergence.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "ht_convergence.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {OUT_DIR / 'ht_convergence.pdf'}")
    print(f"Saved: {OUT_DIR / 'ht_convergence.png'}")

    # ---- Four individual EPS files ----
    save_single_panel_eps(
        plot_density, bern_data, bern_mean, bern_density_title, eps_list,
        OUT_DIR / "ht_bernoulli_density.eps",
    )
    save_single_panel_eps(
        plot_ccdf, bern_data, bern_mean, bern_tail_title, eps_list,
        OUT_DIR / "ht_bernoulli_tail.eps",
    )
    save_single_panel_eps(
        plot_density, burst_data, burst_mean, burst_density_title, eps_list,
        OUT_DIR / "ht_bursty_density.eps",
        aggregate_dx=burst_aggregate_dx,
    )
    save_single_panel_eps(
        plot_ccdf, burst_data, burst_mean, burst_tail_title, eps_list,
        OUT_DIR / "ht_bursty_tail.eps",
    )


if __name__ == "__main__":
    main()
