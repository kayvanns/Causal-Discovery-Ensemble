"""Run one PC or FCI configuration for the CHTC ensemble."""

import argparse
import pickle
import sys
import time
import traceback

import pandas as pd
from causallearn.graph.GraphNode import GraphNode
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz, kci, mv_fisherz, rcit
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

from tiers import ROOT_NODES, get_feature_tier

RUN_CONFIGS = {
    "pc_fisherz": {"algorithm": "PC", "ci_test": fisherz, "use_mvpc": False},
    "pc_mv": {"algorithm": "PC", "ci_test": mv_fisherz, "use_mvpc": True},
    "pc_kci": {"algorithm": "PC", "ci_test": kci, "use_mvpc": False},
    "pc_rcit":     {"algorithm": "PC",  "ci_test": rcit,       "use_mvpc": False},
    "fci_fisherz": {"algorithm": "FCI", "ci_test": fisherz, "use_mvpc": False},
    "fci_mv": {"algorithm": "FCI", "ci_test": mv_fisherz, "use_mvpc": False},
    "fci_kci": {"algorithm": "FCI", "ci_test": kci, "use_mvpc": False},
    "fci_rcit":     {"algorithm": "FCI",  "ci_test": rcit,       "use_mvpc": False},
}

def load_data(path):
    """Load the dataset and its columns."""
    df = pd.read_csv(path)
    col_names = df.columns.tolist()

    untiered = [col for col in col_names if get_feature_tier(col) is None]
    if untiered:
        raise ValueError(f"Columns with no tier assignment: {untiered}")

    print(f"  Rows: {len(df):,}")
    print(f"  Features: {len(col_names)}")
    return df, col_names


def prepare_data(df, sample_size, random_seed=42):
    """Subsample rows and return a float matrix."""
    if 0 < sample_size < len(df):
        df = df.sample(n=sample_size, replace=False, random_state=random_seed)
        print(f"  Subsampled to {sample_size:,} rows")

    return df.to_numpy(dtype=float)


def build_background_knowledge(col_names):
    """Encode temporal tiers and root-node constraints."""
    nodes = [GraphNode(col) for col in col_names]
    col_to_node = dict(zip(col_names, nodes))

    bk = BackgroundKnowledge()
    for col in col_names:
        bk.add_node_to_tier(col_to_node[col], get_feature_tier(col))

    for root in ROOT_NODES:
        if root not in col_to_node:
            continue
        for other in col_names:
            if other != root:
                bk.add_forbidden_by_node(col_to_node[other], col_to_node[root])

    return bk


def run_search(algorithm, data, col_names, config, args, background_knowledge):
    """Dispatch to PC or FCI."""
    if algorithm == "PC":
        cg = pc(
            data=data,
            alpha=args.alpha,
            indep_test=config["ci_test"],
            stable=True,
            mvpc=config["use_mvpc"],
            node_names=col_names,
            background_knowledge=background_knowledge,
            max_k=None if args.depth < 0 else args.depth,
            show_progress=True,
        )
        return cg.G

    graph, _ = fci(
        dataset=data,
        independence_test_method=config["ci_test"],
        alpha=args.alpha,
        depth=args.depth,
        node_names=col_names,
        background_knowledge=background_knowledge,
        verbose=False,
        show_progress=True,
    )
    return graph


def save_graph(graph, col_names, run_name):
    """Pickle the graph and render a PNG."""
    pkl_path = f"{run_name}.pkl"
    with open(pkl_path, "wb") as file:
        pickle.dump((graph, col_names), file)
    print(f"  Saved graph: {pkl_path}")

    png_path = f"{run_name}.png"
    try:
        GraphUtils.to_pydot(graph, labels=col_names).write_png(png_path)
        print(f"  Saved graph image: {png_path}")
    except Exception:
        print(f"  WARNING: PNG rendering failed:\n{traceback.format_exc()}")


def parse_args():
    """Parse the run configuration arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_key", required=True, choices=RUN_CONFIGS.keys())
    parser.add_argument("--run_name", required=True, help="Output prefix.")
    parser.add_argument("--data_path", required=True, help="Input CSV path.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sample_size", type=int, default=-1, help="-1 uses all rows.")
    parser.add_argument("--depth", type=int, default=4, help="-1 is unlimited.")
    return parser.parse_args()


def main():
    """Run one configuration end to end."""
    args = parse_args()
    config = RUN_CONFIGS[args.run_key]
    algorithm = config["algorithm"]
    start_time = time.time()

    print("=" * 80)
    print(f"Run:          {args.run_name}")
    print(f"Run key:      {args.run_key}")
    print(f"Algorithm:    {algorithm}")
    print(f"CI test:      {config['ci_test']}")
    print(f"Alpha:        {args.alpha}")
    print(f"Sample size:  {args.sample_size}")
    print(f"Depth:        {args.depth}")
    print(f"Data:         {args.data_path}")
    print("=" * 80)

    try:
        print("Loading data...")
        df, col_names = load_data(args.data_path)
        data = prepare_data(df, args.sample_size)

        print("Building temporal background knowledge...")
        background_knowledge = build_background_knowledge(col_names)

        print(f"Running {algorithm}...")
        graph = run_search(
            algorithm, data, col_names, config, args, background_knowledge
        )

        save_graph(graph, col_names, args.run_name)
    except Exception:
        print(f"FAILED:\n{traceback.format_exc()}")
        sys.exit(1)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Completed in {elapsed:.1f} seconds ({elapsed / 3600:.2f} hours)")
    print("=" * 80)


if __name__ == "__main__":
    main()
