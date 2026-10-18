"""Summarise the PC/FCI ensemble from the pickles written by causal_discovery.py.
 
Each pickle holds (graph, col_names). Run names follow
{algo}_{test}_{cohort}[_{imputation}], e.g. fci_fisherz_main_knn, pc_mv_main.
 
Outputs (in --out_dir):
  run_summary.csv        size / density / isolated nodes per run
  key_link.csv           exposure-outcome edge in every run, with endpoint marks
  edge_stability.csv     every pair: adjacency frequency per group + modal edge type
  neighbourhood.csv      variables adjacent to exposure / outcome / both
  consensus_main_linear.dot   consensus graph (Graphviz), linear main runs
"""
 
import argparse
import glob
import os
import pickle
from collections import Counter, defaultdict
 
import pandas as pd
 
EXPOSURE = "mechvent_0to24h"
OUTCOME = "aki_24to48h_max_stage"
 
# Sample sizes used for each run 
SAMPLE_SIZES = {
    "pc_rcit_main_knn": 7500,
    "pc_rcit_main_simple": 7500,
}
FCI_DEFAULT_N = 5000
PC_DEFAULT_N = "full"
 
# Endpoint objects are not hashable, so marks are stored as plain strings.
LEFT = {"TAIL": "-", "ARROW": "<", "CIRCLE": "o"}
RIGHT = {"TAIL": "-", "ARROW": ">", "CIRCLE": "o"}
MARK = {"TAIL": "tail", "ARROW": "arrow", "CIRCLE": "circle"}
 
 
def parse_run_name(name):
    parts = name.split("_")
    algo, test, cohort = parts[0].upper(), parts[1], parts[2]
    imputation = parts[3] if len(parts) > 3 else "none (test-wise deletion)"
    family = "nonparametric" if test == "rcit" else "linear"
    n = SAMPLE_SIZES.get(name, FCI_DEFAULT_N if algo == "FCI" else PC_DEFAULT_N)
    return dict(run=name, algorithm=algo, test=test, cohort=cohort,
                imputation=imputation, family=family, n=n)

def load_runs(pkl_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "rb") as f:
            graph, _ = pickle.load(f)
        nodes = [n.get_name() for n in graph.get_nodes()]
        # marks[(a, b)] = (mark at a, mark at b), with a < b alphabetically
        marks = {}
        for e in graph.get_graph_edges():
            a, b = e.get_node1().get_name(), e.get_node2().get_name()
            ea, eb = e.get_endpoint1().name, e.get_endpoint2().name
            if a > b:
                a, b, ea, eb = b, a, eb, ea
            marks[(a, b)] = (ea, eb)
        runs[name] = dict(meta=parse_run_name(name), nodes=set(nodes), marks=marks)
    if not runs:
        raise SystemExit(f"No .pkl files found in {pkl_dir}")
    return runs
 
 
def edge_string(a, b, ea, eb):
    return f"{a} {LEFT[ea]}-{RIGHT[eb]} {b}"
 
 
def oriented(pair, ends, first):
    """Return (x, y, mark_at_x, mark_at_y) with `first` on the left."""
    (a, b), (ea, eb) = pair, ends
    return (a, b, ea, eb) if a == first else (b, a, eb, ea)
 
 
def run_summary(runs):
    rows = []
    for name, r in runs.items():
        n_nodes, n_edges = len(r["nodes"]), len(r["marks"])
        touched = {v for pair in r["marks"] for v in pair}
        MIRROR = {"<--": "-->", "<-o": "o->"}
        types = Counter(MIRROR.get(s, s) for s in
                        (f"{LEFT[ea]}-{RIGHT[eb]}" for ea, eb in r["marks"].values()))
        rows.append({**r["meta"], "nodes": n_nodes, "edges": n_edges,
                     "density": round(2 * n_edges / (n_nodes * (n_nodes - 1)), 3),
                     "isolated_nodes": ", ".join(sorted(r["nodes"] - touched)),
                     **{f"n_{k}": v for k, v in sorted(types.items())}})
    return pd.DataFrame(rows).fillna(0)

def key_link(runs):
    pair = tuple(sorted((EXPOSURE, OUTCOME)))
    rows = []
    for name, r in runs.items():
        row = dict(r["meta"])
        if EXPOSURE not in r["nodes"] or OUTCOME not in r["nodes"]:
            row.update(adjacent="n/a (node missing)")
        elif pair in r["marks"]:
            x, y, ex, ey = oriented(pair, r["marks"][pair], EXPOSURE)
            row.update(adjacent="yes", edge=edge_string(x, y, ex, ey),
                       mark_at_exposure=MARK[ex], mark_at_outcome=MARK[ey])
        else:
            row.update(adjacent="no")
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def group_masks(meta):
    """Which summary groups a run belongs to."""
    groups = [f"{meta['cohort']}_all"]
    if meta["cohort"] == "main":
        groups += [f"main_{meta['family']}", f"main_{meta['algorithm']}"]
    return groups
 
 
def edge_stability(runs):
    all_pairs = set()
    for r in runs.values():
        all_pairs |= set(r["marks"])
 
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # pair -> group -> [hits, denom]
    types = defaultdict(Counter)                                 # pair -> edge strings (main runs)
    for r in runs.values():
        groups = group_masks(r["meta"])
        for pair in all_pairs:
            if not set(pair) <= r["nodes"]:
                continue  # e.g. missingness indicators absent from MV runs
            hit = pair in r["marks"]
            for g in groups:
                counts[pair][g][1] += 1
                counts[pair][g][0] += hit
            if hit and r["meta"]["cohort"] == "main":
                types[pair][edge_string(*pair, *r["marks"][pair])] += 1
 
    rows = []
    for pair in sorted(all_pairs):
        row = {"node_a": pair[0], "node_b": pair[1]}
        for g, (hits, denom) in sorted(counts[pair].items()):
            row[f"{g}_freq"] = round(hits / denom, 2) if denom else None
            row[f"{g}_k/n"] = f"{hits}/{denom}"
        if types[pair]:
            modal, k = types[pair].most_common(1)[0]
            row["modal_edge_main"] = modal
            row["modal_share_main"] = round(k / sum(types[pair].values()), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    sort_col = "main_linear_freq" if "main_linear_freq" in df else df.columns[2]
    return df.sort_values(sort_col, ascending=False)
 
 
def neighbourhood(runs):
    """How often each variable is adjacent to the exposure, the outcome, or both (main runs)."""
    tally = defaultdict(lambda: Counter())
    main = [r for r in runs.values() if r["meta"]["cohort"] == "main"]
    for r in main:
        for v in r["nodes"] - {EXPOSURE, OUTCOME}:
            pe, po = tuple(sorted((v, EXPOSURE))), tuple(sorted((v, OUTCOME)))
            adj_e, adj_o = pe in r["marks"], po in r["marks"]
            tally[v]["denom"] += 1
            tally[v]["exposure"] += adj_e
            tally[v]["outcome"] += adj_o
            tally[v]["both"] += adj_e and adj_o
            if adj_e:
                _, _, _, mark_at_exp = oriented(pe, r["marks"][pe], v)
                tally[v][f"into_exposure_{MARK[mark_at_exp]}"] += 1
    rows = []
    for v, c in tally.items():
        d = c["denom"]
        rows.append({"variable": v, "runs": d,
                     "adj_exposure": round(c["exposure"] / d, 2),
                     "adj_outcome": round(c["outcome"] / d, 2),
                     "adj_both": round(c["both"] / d, 2),
                     "mark_at_exposure_arrow": c["into_exposure_arrow"],
                     "mark_at_exposure_tail": c["into_exposure_tail"],
                     "mark_at_exposure_circle": c["into_exposure_circle"]})
    return pd.DataFrame(rows).sort_values(["adj_both", "adj_outcome"], ascending=False)

def heatmap_table(runs, cohort="main", family=None):
    """Directed edge counts for the heatmap: one row per (cause, effect).
 
    Per run, an edge with an arrowhead at one end only (--> or o->) counts once,
    from the non-arrowhead end to the arrowhead end. Edges with no single
    direction (<->, ---, o-o) count once in BOTH directions, so the heatmap
    cell for each direction shows that the pair was adjacent.
    """
    selected = [r for r in runs.values() if r["meta"]["cohort"] == cohort
                and (family is None or r["meta"]["family"] == family)]
    directed, unoriented = Counter(), Counter()
    for r in selected:
        for (a, b), (ea, eb) in r["marks"].items():
            if eb == "ARROW" and ea != "ARROW":
                directed[(a, b)] += 1
            elif ea == "ARROW" and eb != "ARROW":
                directed[(b, a)] += 1
            else:
                unoriented[(a, b)] += 1
                unoriented[(b, a)] += 1
    rows = []
    for cause, effect in set(directed) | set(unoriented):
        d, u = directed[(cause, effect)], unoriented[(cause, effect)]
        rows.append({"edge": f"{cause} -> {effect}", "cause": cause, "effect": effect,
                     "agreement_score": d + u, "n_directed": d, "n_unoriented": u,
                     "n_runs": len(selected)})
    return pd.DataFrame(rows).sort_values("agreement_score", ascending=False)
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_dir", default="results/graphs")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
 
    runs = load_runs(args.pkl_dir)
    print(f"Loaded {len(runs)} runs: {', '.join(runs)}\n")
 
    rs = run_summary(runs)
    kl = key_link(runs)
    st = edge_stability(runs)
    nb = neighbourhood(runs)
 
    rs.to_csv(f"{args.out_dir}/run_summary.csv", index=False)
    kl.to_csv(f"{args.out_dir}/key_link.csv", index=False)
    st.to_csv(f"{args.out_dir}/edge_stability.csv", index=False)
    nb.to_csv(f"{args.out_dir}/neighbourhood.csv", index=False)
    heatmap_table(runs).to_csv(f"{args.out_dir}/heatmap_table.csv", index=False)
 
    print("KEY LINK:", EXPOSURE, "--", OUTCOME)
    print(kl[["run", "family", "cohort", "n", "adjacent", "mark_at_exposure",
              "mark_at_outcome"]].to_string(index=False), "\n")
    print("RUN SIZES")
    print(rs[["run", "edges", "density", "isolated_nodes"]].to_string(index=False))
    print(f"\nWrote tables to {args.out_dir}/")
 
 
if __name__ == "__main__":
    main()