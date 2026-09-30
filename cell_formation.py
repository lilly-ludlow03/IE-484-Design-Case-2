import math, random, itertools
import numpy as np
import pulp
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# routing data: part -> [(machine type, hours per unit), ...]
ROUTING = {
    "A": [(4, .36), (7, .25), (8, .17)],
    "B": [(1, .30), (2, .20), (3, .30)],
    "C": [(2, .26), (5, .16), (10, .27)],
    "D": [(1, .33), (6, .24), (8, .13)],
    "E": [(1, .27), (3, .15), (7, .46), (8, .30)],
    "F": [(3, .15), (4, .41), (5, .22), (6, .24)],
    "G": [(2, .24), (6, .25), (8, .36)],
    "H": [(3, .32), (4, .12), (10, .28)],
    "I": [(4, .42), (6, .51), (7, .31), (9, .40)],
    "J": [(4, .40), (5, .29), (6, .28)],
    "K": [(3, .28), (5, .24), (10, .24)],
    "L": [(1, .36), (4, .16), (5, .34), (8, .36)],
    "M": [(4, .12), (5, .10), (7, .32), (8, .22)],
}

# machines owned
OWNED = {1: 5, 2: 3, 3: 5, 4: 7, 5: 6, 6: 6, 7: 6, 8: 6, 9: 3, 10: 3}

# product data: annual demand and components per unit
PRODUCTS = {
    1: {"demand": 4245, "parts": ["A", "K", "L", "M"]},
    2: {"demand": 4590, "parts": ["B", "E", "F", "H", "J"]},
    3: {"demand": 3400, "parts": ["C", "D", "G", "I"]},
}

# capacity assumptions
HOURS_PER_WEEK = 40
SETUP_DOWNTIME = 1            # setup and tear-down
WEEKS_PER_YEAR = 50           # holidays, time off
TARGET_UTIL = 0.74
MACHINE_CAPACITY = 0.92
MAX_CELLS = 5                  # maximum number of cells allowed
MAX_MACHINES_PER_CELL = 100    # cell size limit (as many machines as needed per cell)

# weights for the MILP
ALPHA = 5.0      # per 1000 units of inter-cell operations
BETA = 2.0       # per machine copy placed
GAMMA = 0.002    # per hour of workload in the busiest cell
DELTA = 5.0      # per cell opened

# similarity weights
W_JACCARD, W_SEQ, W_LOAD = 0.4, 0.4, 0.2

PARTS = sorted(ROUTING)
MACHINES = sorted(OWNED)
T = {p: {m: t for m, t in ROUTING[p]} for p in PARTS}
SEQ = {p: [m for m, _ in ROUTING[p]] for p in PARTS}


def part_demand(scale=1.0, mix=None):
    d = {}
    for pid, info in PRODUCTS.items():
        f = scale * (mix.get(pid, 1.0) if mix else 1.0)
        for p in info["parts"]:
            d[p] = info["demand"] * f
    return d


def effective_hours():
    """Productive hours per machine per year."""
    return (HOURS_PER_WEEK - SETUP_DOWNTIME) * WEEKS_PER_YEAR * MACHINE_CAPACITY * TARGET_UTIL


def workloads(D):
    return {m: sum(D[p] * T[p][m] for p in PARTS if m in T[p]) for m in MACHINES}


def stage0_report(D):
    H = effective_hours()
    W = workloads(D)
    print("=== STAGE 0: workload and machine requirements ===")
    print(f"Effective hours per machine per year: {H:,.1f}")
    print(f"{'Mach':>4} {'Load(hr)':>10} {'Needed':>7} {'Ceil':>5} {'Owned':>6}")
    for m in MACHINES:
        r = W[m] / H
        flag = "  <-- SHORT" if math.ceil(r - 1e-9) > OWNED[m] else ""
        print(f"{m:>4} {W[m]:>10,.0f} {r:>7.2f} {math.ceil(r-1e-9):>5} {OWNED[m]:>6}{flag}")
    print()


# Similarity matrix
def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            dp[i+1][j+1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j+1], dp[i+1][j])
    return dp[-1][-1]


def similarity_matrix():
    n = len(PARTS)
    S = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        p, q = PARTS[i], PARTS[j]
        Mp, Mq = set(SEQ[p]), set(SEQ[q])
        jac = len(Mp & Mq) / len(Mp | Mq)
        seq = lcs(SEQ[p], SEQ[q]) / max(len(SEQ[p]), len(SEQ[q]))
        common = Mp & Mq
        tp, tq = sum(T[p].values()), sum(T[q].values())
        load = sum(T[p][m] + T[q][m] for m in common) / (tp + tq)
        S[i, j] = S[j, i] = W_JACCARD * jac + W_SEQ * seq + W_LOAD * load
    return S


# Groupings and grouping efficacy
def grouping_efficacy(assign):
    """assign: part -> cell. Each machine type goes to the cell using it most."""
    cells = sorted(set(assign.values()))
    e = e0 = ev = 0
    for m in MACHINES:
        users = [p for p in PARTS if m in T[p]]
        counts = {c: sum(1 for p in users if assign[p] == c) for c in cells}
        home = max(counts, key=counts.get)
        e += len(users)
        e0 += len(users) - counts[home]
        ev += sum(1 for p in PARTS if assign[p] == home and m not in T[p])
    return (e - e0) / (e + ev), e0, ev


def stage2_candidates(S):
    dist = squareform(1 - S, checks=False)
    Z = linkage(dist, method="average")
    out = []
    for k in range(2, MAX_CELLS + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        assign = {p: int(labels[i]) - 1 for i, p in enumerate(PARTS)}
        eff, e0, ev = grouping_efficacy(assign)
        out.append((eff, k, e0, ev, assign))
    out.sort(key=lambda r: -r[0])
    print("=== STAGE 2: candidate groupings (average-linkage on composite similarity) ===")
    for eff, k, e0, ev, a in out:
        groups = {}
        for p, c in a.items():
            groups.setdefault(c, []).append(p)
        print(f"k={k}  efficacy={eff:.3f}  exceptional={e0}  voids={ev}  "
              + " | ".join("".join(v) for v in groups.values()))
    print()
    return out


# MILP
def solve_milp(D, warm=None, time_limit=120, verbose=True, owned=None):
    owned = owned or OWNED
    H = effective_hours()
    C = range(MAX_CELLS)
    prob = pulp.LpProblem("cell_formation", pulp.LpMinimize)

    x = {(p, c): pulp.LpVariable(f"x_{p}_{c}", cat="Binary") for p in PARTS for c in C}
    u = {c: pulp.LpVariable(f"u_{c}", cat="Binary") for c in C}
    y = {(m, c): pulp.LpVariable(f"y_{m}_{c}", 0, owned[m], cat="Integer") for m in MACHINES for c in C}
    f = {(p, m, c): pulp.LpVariable(f"f_{p}_{m}_{c}", 0, 1) for p in PARTS for m in SEQ[p] for c in C}
    out = {(p, m, c): pulp.LpVariable(f"o_{p}_{m}_{c}", 0, 1) for p in PARTS for m in SEQ[p] for c in C}
    Lmax = pulp.LpVariable("Lmax", 0)

    inter = pulp.lpSum(D[p] / 1000 * out[p, m, c] for p in PARTS for m in SEQ[p] for c in C)
    prob += (ALPHA * inter + BETA * pulp.lpSum(y.values())
             + GAMMA * Lmax + DELTA * pulp.lpSum(u.values()))

    for p in PARTS:
        prob += pulp.lpSum(x[p, c] for c in C) == 1
        for c in C:
            prob += x[p, c] <= u[c]
        for m in SEQ[p]:
            prob += pulp.lpSum(f[p, m, c] for c in C) == 1
            for c in C:
                prob += out[p, m, c] >= f[p, m, c] - x[p, c]     # work done away from home cell
                prob += f[p, m, c] <= y[m, c]                     # need a machine there
    for m in MACHINES:
        prob += pulp.lpSum(y[m, c] for c in C) <= owned[m]
        for c in C:
            prob += pulp.lpSum(D[p] * T[p][m] * f[p, m, c] for p in PARTS if m in T[p]) <= H * y[m, c]
            prob += y[m, c] <= owned[m] * u[c]
    for c in C:
        prob += pulp.lpSum(y[m, c] for m in MACHINES) <= MAX_MACHINES_PER_CELL
        prob += Lmax >= pulp.lpSum(D[p] * sum(T[p].values()) * x[p, c] for p in PARTS)
        if c < MAX_CELLS - 1:
            prob += u[c] >= u[c + 1]

    if warm:
        used = sorted(set(warm.values()))
        remap = {c: i for i, c in enumerate(used)}
        for p in PARTS:
            for c in C:
                x[p, c].setInitialValue(1 if remap[warm[p]] == c else 0)
        for c in C:
            u[c].setInitialValue(1 if c < len(used) else 0)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, warmStart=bool(warm))
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status != "Optimal" and not any(v.value() for v in x.values()):
        return None, status

    home = {p: next(c for c in C if x[p, c].value() > 0.5) for p in PARTS}
    cells_used = sorted(set(home.values()))
    remap = {c: i + 1 for i, c in enumerate(cells_used)}
    sol = {
        "status": status,
        "objective": pulp.value(prob.objective),
        "home": {p: remap[c] for p, c in home.items()},
        "y": {(m, remap[c]): int(round(y[m, c].value())) for m in MACHINES for c in cells_used
              if y[m, c].value() and y[m, c].value() > 0.5},
        "f": {(p, m, remap[c]): f[p, m, c].value() for (p, m, c) in f
              if c in remap and f[p, m, c].value() > 1e-6},
        "inter_units": sum(D[p] * out[p, m, c].value() for (p, m, c) in out),
    }
    return sol, status


def print_solution(sol, D):
    H = effective_hours()
    cells = sorted(set(sol["home"].values()))
    print("=== STAGE 3: MILP solution ===")
    print(f"Status: {sol['status']}   Objective: {sol['objective']:.2f}   "
          f"Cells: {len(cells)}   Inter-cell operation-units: {sol['inter_units']:,.0f}")
    total_machines = 0
    for c in cells:
        parts = [p for p in PARTS if sol["home"][p] == c]
        print(f"\nCell {c}: parts {', '.join(parts)}")
        print(f"  {'Mach':>4} {'Copies':>6} {'Load(hr)':>10} {'Util vs target-hrs':>19}")
        cell_count = 0
        for m in MACHINES:
            n = sol["y"].get((m, c), 0)
            if not n:
                continue
            load = sum(D[p] * T[p][m] * sol["f"].get((p, m, c), 0) for p in PARTS if m in T[p])
            print(f"  {m:>4} {n:>6} {load:>10,.0f} {load / (H * n):>18.0%}")
            cell_count += n
        print(f"  Machines in cell: {cell_count}")
        total_machines += cell_count
    print(f"\nTotal machines used: {total_machines} of {sum(OWNED.values())} owned")
    split = [(p, m) for p in PARTS for m in SEQ[p]
             if sum(1 for c in cells if sol["f"].get((p, m, c), 0) > 1e-6) > 1
             or sol["f"].get((p, m, sol["home"][p]), 0) < 1 - 1e-6]
    if split:
        print("Operations done outside (or split from) the home cell:")
        for p, m in split:
            parts_txt = ", ".join(f"cell {c}: {sol['f'][p, m, c]:.0%}" for c in cells
                                  if sol["f"].get((p, m, c), 0) > 1e-6)
            print(f"  part {p}, machine {m} -> {parts_txt}")
    else:
        print("No exceptional elements: every operation is done in its home cell.")
    print()


# QAP for cell layout
def u_slots(n):
    """Slot coordinates on a U: down the left arm, up the right arm."""
    k = math.ceil(n / 2)
    pts = [(0, i) for i in range(k)]
    pts += [(1, k - 1 - j) for j in range(n - k)]
    return pts


def layout_cell(c, sol, D, seed=0):
    types = sorted({m for (m, cc) in sol["y"] if cc == c})
    n = len(types)
    idx = {m: i for i, m in enumerate(types)}
    flow = np.zeros((n, n))
    for p in PARTS:
        for a, b in zip(SEQ[p], SEQ[p][1:]):
            fa, fb = sol["f"].get((p, a, c), 0), sol["f"].get((p, b, c), 0)
            if fa > 0 and fb > 0:
                flow[idx[a], idx[b]] += D[p] * min(fa, fb)
    slots = u_slots(n)
    dist = np.array([[abs(a[0] - b[0]) * 1.5 + abs(a[1] - b[1]) for b in slots] for a in slots])

    def cost(perm):
        return sum(flow[i, j] * dist[perm[i], perm[j]] for i in range(n) for j in range(n) if flow[i, j])

    rng = random.Random(seed)
    best, best_cost = None, float("inf")
    for _ in range(30):
        perm = list(range(n))
        rng.shuffle(perm)
        cur = cost(perm)
        improved = True
        while improved:
            improved = False
            for i, j in itertools.combinations(range(n), 2):
                perm[i], perm[j] = perm[j], perm[i]
                new = cost(perm)
                if new < cur - 1e-9:
                    cur, improved = new, True
                else:
                    perm[i], perm[j] = perm[j], perm[i]
        if cur < best_cost:
            best, best_cost = perm[:], cur
    order = sorted(range(n), key=lambda i: best[i])       # machine types by slot number
    layout = [types[i] for i in order]
    # flow that moves to an earlier slot around the U (higher slot -> lower slot)
    slot_of = {types[i]: best[i] for i in range(n)}
    back = sum(flow[i, j] for i in range(n) for j in range(n) if best[i] > best[j])
    total = flow.sum()
    return layout, best_cost, back / total if total else 0


def stage5(sol, D):
    print("=== STAGE 5: U-shape layouts (machine types in slot order along the U) ===")
    for c in sorted(set(sol["home"].values())):
        layout, cost, back = layout_cell(c, sol, D)
        k = math.ceil(len(layout) / 2)
        left, right = layout[:k], layout[k:][::-1]
        print(f"Cell {c}: entry -> {' -> '.join(f'M{m}' for m in layout)}")
        print(f"   left arm  (top->bottom): {left}")
        print(f"   right arm (top->bottom): {right}")
        print(f"   flow-distance cost {cost:,.0f}, backtracking {back:.0%} of moves\n")

# Validating results
def stage6(D):
    print("=== STAGE 6: robustness ===")
    H = effective_hours()
    print("Demand sensitivity (all products scaled):")
    for scale in (0.8, 1.0, 1.2):
        Ds = part_demand(scale)
        W = workloads(Ds)
        short = [m for m in MACHINES if math.ceil(W[m] / H - 1e-9) > OWNED[m]]
        if short:
            print(f"  x{scale}: INFEASIBLE with owned machines, short on types {short}")
            continue
        sol, st = solve_milp(Ds, time_limit=40)
        if sol:
            print(f"  x{scale}: {len(set(sol['home'].values()))} cells, "
                  f"machines {sum(sol['y'].values())}, inter-cell units {sol['inter_units']:,.0f}")
        else:
            print(f"  x{scale}: no solution ({st})")
    print("\nMachine-loss test (lose one copy of a type, is demand still coverable?):")
    W = workloads(D)
    for m in MACHINES:
        need = W[m] / H
        ok = need <= OWNED[m] - 1
        print(f"  M{m}: need {need:.2f} copies, own {OWNED[m]} -> "
              + ("survives losing one" if ok else "NO spare capacity"))
    print()

def main():
    D = part_demand()
    stage0_report(D)
    S = similarity_matrix()
    cands = stage2_candidates(S)
    best_assign = cands[0][4]
    sol, status = solve_milp(D, warm=best_assign)
    if not sol:
        print("MILP found no solution:", status)
        return
    print_solution(sol, D)
    stage5(sol, D)
    stage6(D)


if __name__ == "__main__":
    main()
