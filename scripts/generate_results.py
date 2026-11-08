"""Generate the validation report, result tables and figures into docs/results/.

Usage (from the project root):  python scripts/generate_results.py
Everything uses fixed seeds, so the numbers are reproducible.
PNG images need the optional `kaleido` package (and Google Chrome); HTML versions are always written.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optionlab.convergence import binomial_convergence, monte_carlo_convergence, monte_carlo_single_runs
from optionlab.greeks import fd_greeks, greeks, mc_delta
from optionlab.pricing.binomial import binomial_price
from optionlab.pricing.black_scholes import call_price, put_price
from optionlab.pricing.monte_carlo import mc_price
from optionlab.simulation.scenario import simulate_scenario
from optionlab.simulation.stats import gbm_quantile
from optionlab.validation import run_validation, to_markdown

OUT = Path("docs") / "results"
A = dict(S=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def _save(fig, out_dir, name, images):
    fig.write_html(out_dir / f"{name}.html", include_plotlyjs="cdn")
    if not images:
        return False
    try:
        fig.write_image(out_dir / f"{name}.png", width=1300, height=520, scale=2)
        return True
    except Exception as exc:        # kaleido or Chrome missing: keep going with HTML only
        print(f"  PNG for {name} skipped ({type(exc).__name__}); HTML written instead.")
        return False


def pricing_table(quick):
    bc, bp = call_price(**A), put_price(**A)
    rows = [{"method": "Black-Scholes (exact)", "call": bc, "call - BS": 0.0, "call SE": 0.0,
             "put": bp, "put - BS": 0.0}]
    for n in (100, 500, 2000):
        c, p = binomial_price(n=n, **A), binomial_price(n=n, option_type="put", **A)
        rows.append({"method": f"Binomial tree, n = {n:,}", "call": c, "call - BS": c - bc,
                     "call SE": 0.0, "put": p, "put - BS": p - bp})
    sizes = ((10_000, False), (100_000, False), (100_000, True)) if quick else \
            ((10_000, False), (100_000, False), (100_000, True), (1_000_000, True))
    for n, anti in sizes:
        c = mc_price(**A, n_paths=n, seed=42, antithetic=anti)
        p = mc_price(**A, n_paths=n, option_type="put", seed=43, antithetic=anti)
        label = f"Monte Carlo, N = {n:,}" + (", antithetic" if anti else "")
        rows.append({"method": label, "call": c.price, "call - BS": c.price - bc,
                     "call SE": c.std_error, "put": p.price, "put - BS": p.price - bp})
    return pd.DataFrame(rows)


def greeks_table():
    rows = []
    for option_type in ("call", "put"):
        a, f = greeks(**A, option_type=option_type), fd_greeks(**A, option_type=option_type)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            rows.append({"option": option_type, "greek": name, "analytic": getattr(a, name),
                         "finite difference": getattr(f, name),
                         "abs difference": abs(getattr(a, name) - getattr(f, name))})
    est, se = mc_delta(**A, n_paths=200_000, seed=42)
    rows.append({"option": "call", "greek": "delta (Monte Carlo, pathwise)", "analytic": greeks(**A).delta,
                 "finite difference": est, "abs difference": abs(est - greeks(**A).delta)})
    return pd.DataFrame(rows)


def figures(out_dir, quick, images):
    made = []
    bc = binomial_convergence(**A, n_values=range(5, 201))
    ns = sorted({n for g in np.unique(np.round(np.geomspace(10, 500 if quick else 2000, 20)).astype(int))
                 for n in (int(g), int(g) + 1)})
    wide = binomial_convergence(**A, n_values=ns)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Binomial price vs n", "|error| vs n (log-log)"))
    fig.add_trace(go.Scatter(x=bc["n"], y=bc["binomial"], mode="lines+markers", name="binomial"), row=1, col=1)
    fig.add_hline(y=call_price(**A), line_dash="dash", annotation_text="Black-Scholes", row=1, col=1)
    fig.add_trace(go.Scatter(x=wide["n"], y=wide["abs_error"], mode="markers", name="|error|"), row=1, col=2)
    fig.add_trace(go.Scatter(x=wide["n"], y=wide["abs_error"].iloc[0] * wide["n"].iloc[0] / wide["n"],
                             mode="lines", name="c/n", line=dict(dash="dash")), row=1, col=2)
    fig.update_xaxes(title_text="n", row=1, col=1)
    fig.update_xaxes(title_text="n", type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    made.append(("fig_binomial_convergence", fig))

    counts = (100, 1_000, 10_000, 100_000) if quick else (100, 1_000, 10_000, 100_000, 1_000_000)
    reps = 5 if quick else 20
    plain = monte_carlo_convergence(**A, path_counts=counts, replications=reps)
    anti = monte_carlo_convergence(**A, path_counts=counts, replications=reps, antithetic=True)
    runs = monte_carlo_single_runs(**A, path_counts=counts, antithetic=True)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("One run per size, 95% interval", "RMSE vs N (log-log)"))
    fig.add_trace(go.Scatter(x=runs["n_paths"], y=runs["price"], mode="markers", name="MC price",
                             error_y=dict(type="data", symmetric=False, array=runs["ci_high"] - runs["price"],
                                          arrayminus=runs["price"] - runs["ci_low"])), row=1, col=1)
    fig.add_hline(y=call_price(**A), line_dash="dash", annotation_text="Black-Scholes", row=1, col=1)
    fig.add_trace(go.Scatter(x=plain["n_paths"], y=plain["rmse"], mode="lines+markers", name="plain"), row=1, col=2)
    fig.add_trace(go.Scatter(x=anti["n_paths"], y=anti["rmse"], mode="lines+markers", name="antithetic"), row=1, col=2)
    n = plain["n_paths"].to_numpy(dtype=float)
    fig.add_trace(go.Scatter(x=n, y=plain["rmse"].iloc[0] * np.sqrt(n[0] / n), mode="lines",
                             name="c/sqrt(N)", line=dict(dash="dash")), row=1, col=2)
    fig.update_xaxes(type="log", title_text="paths N")
    fig.update_yaxes(type="log", row=1, col=2)
    made.append(("fig_monte_carlo_convergence", fig))

    sc = simulate_scenario(100.0, 0.08, 0.2, 1.0, n_steps=252, n_paths=20_000, n_draw=100, seed=42)
    t = sc.times
    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
                        subplot_titles=("100 simulated GBM paths (real-world, mu = 8%)", "Terminal price"))
    rows = sc.sample_paths.shape[0]
    x = np.tile(np.append(t, np.nan), rows)
    y = np.column_stack([sc.sample_paths, np.full(rows, np.nan)]).ravel()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", opacity=0.4, line=dict(width=1), name="paths"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=gbm_quantile(100.0, 0.08, 0.2, t, 0.05), mode="lines",
                             line=dict(dash="dash"), name="5% quantile"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=gbm_quantile(100.0, 0.08, 0.2, t, 0.95), mode="lines",
                             line=dict(dash="dash"), name="95% quantile"), row=1, col=1)
    fig.add_trace(go.Histogram(x=sc.terminal_prices, nbinsx=80, name="S_T"), row=1, col=2)
    made.append(("fig_gbm_paths", fig))

    spots = list(range(50, 151))
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Delta", "Gamma", "Vega", "Theta (per year)"))
    for i, field in enumerate(("delta", "gamma", "vega", "theta")):
        for option_type in ("call", "put"):
            ys = [getattr(greeks(s, A["K"], A["r"], A["sigma"], A["T"], option_type), field) for s in spots]
            fig.add_trace(go.Scatter(x=spots, y=ys, mode="lines", name=option_type, legendgroup=option_type,
                                     showlegend=(i == 0)), row=i // 2 + 1, col=i % 2 + 1)
    fig.update_xaxes(title_text="spot price")
    made.append(("fig_greeks", fig))

    saved = {}
    for name, fig in made:
        fig.update_layout(template="plotly_white", legend=dict(orientation="h", y=-0.2))
        saved[name] = _save(fig, out_dir, name, images)
    return saved


def main(out_dir=OUT, quick=False, images=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Running validation checks...")
    validation = run_validation()
    pricing = pricing_table(quick)
    binomial = binomial_convergence(**A, n_values=(10, 50, 100, 500, 1000, 2000))
    counts = (100, 1_000, 10_000, 100_000) if quick else (100, 1_000, 10_000, 100_000, 1_000_000)
    reps = 5 if quick else 20
    p = monte_carlo_convergence(**A, path_counts=counts, replications=reps)
    q = monte_carlo_convergence(**A, path_counts=counts, replications=reps, antithetic=True)
    mc = pd.DataFrame({"N paths": p["n_paths"], "plain: average SE": p["mean_std_error"], "plain: RMSE": p["rmse"],
                       "antithetic: average SE": q["mean_std_error"], "antithetic: RMSE": q["rmse"]})
    gk = greeks_table()
    tables = {"validation": validation, "pricing_comparison": pricing, "binomial_convergence": binomial,
              "monte_carlo_convergence": mc, "greeks": gk}
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    print("Drawing figures...")
    saved = figures(out_dir, quick, images)

    passed = int((validation["status"] == "PASS").sum())
    lines = [
        "# Results", "",
        "Generated by `python scripts/generate_results.py` (fixed random seeds, so reproducible).",
        "Option: European, S = K = 100, r = 5%, sigma = 20%, T = 1 year (no dividends).", "",
        f"## Validation: {passed} of {len(validation)} checks pass", "", to_markdown(validation), "",
        "## The three methods side by side", "", to_markdown(pricing.round(5)), "",
        "## Binomial convergence", "", to_markdown(binomial.round(5)), "",
        "## Monte Carlo convergence", "",
        f"Each size repeated {reps} times; RMSE is the typical actual error.", "", to_markdown(mc.round(5)), "",
        "## Greeks", "", to_markdown(gk.round(6)), "",
        "## Figures", "",
    ]
    for name, has_png in saved.items():
        lines.append(f"![{name}]({name}.png)" if has_png else f"- [{name}]({name}.html) (interactive)")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Done: {passed} of {len(validation)} validation checks pass. Files are in {out_dir}/")
    return tables


if __name__ == "__main__":
    main()
