import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cell_formation import PRODUCTS, OWNED, MACHINES, part_demand, workloads, effective_hours

GROW_YEARS = 5
TOTAL_YEARS = 8
GROWTH = (0.11, 0.18)       # per-year growth range, years 1-5
DECLINE = (0.04, 0.09)      # per-year decline range, years 6-8
N_SIM = 10_000
SEED = 42

H = effective_hours()       # productive hours per machine per year

def rates_for(growth, decline):
    """Signed yearly rate list (length 8) from one growth and one decline value."""
    return [growth] * GROW_YEARS + [-decline] * (TOTAL_YEARS - GROW_YEARS)


def demand_index(rates):
    """Cumulative demand multiplier for years 0..8 (year 0 = 1.0)."""
    idx = [1.0]
    for r in rates:
        idx.append(idx[-1] * (1 + r))
    return np.array(idx)


def machines_needed(scale):
    """Fractional machines required per type at a given demand multiplier."""
    W = workloads(part_demand(scale))
    return np.array([W[m] / H for m in MACHINES])


def ceil_req(x):
    return np.ceil(x - 1e-9).astype(int)


OWN = np.array([OWNED[m] for m in MACHINES])


def scenario_table(rates):
    idx = demand_index(rates)
    need = np.array([machines_needed(s) for s in idx])   # (9, 10)
    return idx, need

SCENARIOS = {
    "LOW  (11'%' growth, 9'%' decline)": rates_for(GROWTH[0], DECLINE[1]),
    "MID  (14.5'%' growth, 6.5'%' decline)": rates_for(sum(GROWTH) / 2, sum(DECLINE) / 2),
    "HIGH (18'%' growth, 4'%' decline)": rates_for(GROWTH[1], DECLINE[0]),
}


def print_scenario(name, rates):
    idx, need = scenario_table(rates)
    req = np.array([ceil_req(n) for n in need])
    print(f"\n{'=' * 100}\n{name}\n{'=' * 100}")
    print(f"Effective hours per machine per year: {H:,.1f}")
    print("\nProduct demand (units/year):")
    print(f"{'Year':>5} {'Rate':>7} " + " ".join(f"{'Prod ' + str(k):>9}" for k in PRODUCTS))
    for y in range(TOTAL_YEARS + 1):
        rate = "-" if y == 0 else f"{rates[y - 1]:+.1%}"
        print(f"{y:>5} {rate:>7} " + " ".join(f"{v['demand'] * idx[y]:>9,.0f}" for v in PRODUCTS.values()))

    print("\nMachines required (rounded up)   [* = exceeds machines owned]")
    print(f"{'Year':>5} " + " ".join(f"{'M' + str(m):>5}" for m in MACHINES) + f" {'Total':>6} {'Owned':>6} {'Buy':>4}")
    for y in range(TOTAL_YEARS + 1):
        cells = []
        for j in range(len(MACHINES)):
            s = f"{req[y, j]}" + ("*" if req[y, j] > OWN[j] else "")
            cells.append(f"{s:>5}")
        buy = int(np.maximum(req[y] - OWN, 0).sum())
        print(f"{y:>5} " + " ".join(cells) + f" {req[y].sum():>6} {OWN.sum():>6} {buy:>4}")
    peak_year = int(np.argmax(req.sum(axis=1)))
    extra = np.maximum(req.max(axis=0) - OWN, 0)
    print(f"\nPeak demand year: {peak_year} (index {idx[peak_year]:.2f}x today)")
    if extra.sum():
        print("Extra machines needed at peak: " +
              ", ".join(f"M{m} +{e}" for m, e in zip(MACHINES, extra) if e))
    else:
        print("Owned machines cover the whole horizon.")
    first = {}
    for j, m in enumerate(MACHINES):
        ys = np.where(req[:, j] > OWN[j])[0]
        if len(ys):
            first[m] = int(ys[0])
    if first:
        print("First year each type runs short: " + ", ".join(f"M{m}: year {y}" for m, y in first.items()))
    return idx, need, req


# Monte Carlo
def monte_carlo():
    rng = np.random.default_rng(SEED)
    g = rng.uniform(*GROWTH, size=(N_SIM, GROW_YEARS))
    d = rng.uniform(*DECLINE, size=(N_SIM, TOTAL_YEARS - GROW_YEARS))
    rates = np.hstack([g, -d])                                     # (N, 8)
    idx = np.hstack([np.ones((N_SIM, 1)), np.cumprod(1 + rates, axis=1)])   # (N, 9)
    base = machines_needed(1.0)                                    # need is linear in demand
    need = idx[:, :, None] * base[None, None, :]                   # (N, 9, 10)
    req = np.ceil(need - 1e-9).astype(int)
    return idx, req


def print_mc(idx, req):
    print(f"\n{'=' * 100}\nMONTE CARLO ({N_SIM:,} runs, each year's rate drawn uniformly from its range)\n{'=' * 100}")
    tot = req.sum(axis=2)
    print(f"{'Year':>5} {'Demand idx P10':>15} {'P50':>7} {'P90':>7} | "
          f"{'Machines P10':>12} {'P50':>5} {'P90':>5} | {'P(any type short)':>18}")
    for y in range(TOTAL_YEARS + 1):
        di = np.percentile(idx[:, y], [10, 50, 90])
        mt = np.percentile(tot[:, y], [10, 50, 90])
        short_any = (req[:, y, :] > OWN).any(axis=1).mean()
        print(f"{y:>5} {di[0]:>15.2f} {di[1]:>7.2f} {di[2]:>7.2f} | "
              f"{mt[0]:>12.0f} {mt[1]:>5.0f} {mt[2]:>5.0f} | {short_any:>18.0%}")
    print("\nProbability each machine type is short of owned copies, by year:")
    print(f"{'Year':>5} " + " ".join(f"{'M' + str(m):>5}" for m in MACHINES))
    for y in range(TOTAL_YEARS + 1):
        p = (req[:, y, :] > OWN).mean(axis=0)
        print(f"{y:>5} " + " ".join(f"{v:>5.0%}" for v in p))
    p90_peak = req.max(axis=1)                                     # (N, 10) peak over horizon
    buy90 = np.percentile(np.maximum(p90_peak - OWN, 0), 90, axis=0).astype(int)
    print("\nExtra copies needed to be 90'%' safe over the full horizon: " +
          (", ".join(f"M{m} +{b}" for m, b in zip(MACHINES, buy90) if b) or "none"))


# output
def save_csv(results, path="demand_forecast_scenarios.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "year", "demand_index"] +
                   [f"prod{k}_units" for k in PRODUCTS] +
                   [f"M{m}_required" for m in MACHINES] + ["total_required", "total_owned"])
        for name, (idx, need, req) in results.items():
            for y in range(TOTAL_YEARS + 1):
                w.writerow([name.split()[0], y, round(idx[y], 4)] +
                           [round(v["demand"] * idx[y]) for v in PRODUCTS.values()] +
                           list(req[y]) + [int(req[y].sum()), int(OWN.sum())])
    print(f"\nSaved {path}")


def plot(results, mc_idx, mc_req, path="demand_forecast.png"):
    years = np.arange(TOTAL_YEARS + 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    colors = {"LOW": "tab:green", "MID": "tab:blue", "HIGH": "tab:red"}

    lo, md, hi = (np.percentile(mc_idx, q, axis=0) for q in (10, 50, 90))
    ax[0].fill_between(years, lo, hi, color="grey", alpha=0.25, label="Monte Carlo P10-P90")
    ax[0].plot(years, md, color="k", lw=1, ls=":", label="Monte Carlo median")
    for name, (idx, _, _) in results.items():
        k = name.split()[0]
        ax[0].plot(years, idx, marker="o", color=colors[k], label=k)
    ax[0].axvline(GROW_YEARS, color="grey", ls="--", lw=0.8)
    ax[0].set(title="Demand index (year 0 = 1.0)", xlabel="Year")
    ax[0].legend(fontsize=8)

    tot = mc_req.sum(axis=2)
    ax[1].fill_between(years, np.percentile(tot, 10, axis=0), np.percentile(tot, 90, axis=0),
                       color="grey", alpha=0.25, label="Monte Carlo P10-P90")
    for name, (_, _, req) in results.items():
        k = name.split()[0]
        ax[1].plot(years, req.sum(axis=1), marker="o", color=colors[k], label=k)
    ax[1].axhline(OWN.sum(), color="k", ls="--", label=f"Owned ({OWN.sum()})")
    ax[1].set(title="Total machines required", xlabel="Year")
    ax[1].legend(fontsize=8)

    mid = results[[k for k in results if k.startswith("MID")][0]][2]
    for j, m in enumerate(MACHINES):
        ax[2].plot(years, mid[:, j] - OWN[j], marker=".", label=f"M{m}")
    ax[2].axhline(0, color="k", lw=1)
    ax[2].set(title="MID scenario: required minus owned (>0 = shortage)", xlabel="Year")
    ax[2].legend(ncol=2, fontsize=7)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"Saved {path}")


def main():
    results = {}
    for name, rates in SCENARIOS.items():
        results[name] = print_scenario(name, rates)
    mc_idx, mc_req = monte_carlo()
    print_mc(mc_idx, mc_req)
    save_csv(results)
    plot(results, mc_idx, mc_req)


if __name__ == "__main__":
    main()