"""Evidence for the missingness-indicator groupings declared in tiers.py.

Every indicator group in tiers.INDICATOR_GROUPS collapses several columns onto
one missingness indicator on the grounds that they are ordered as a unit, so
their individual indicators are near-duplicates rather than separate evidence.
This script is where the r values quoted in tiers.py and datasets.py come from:
it recomputes the full pairwise correlation matrix of the individual indicators
on the analysis cohort and reports, per group, the within-group correlations
that justify collapsing and the cross-group correlations that justify keeping
the groups apart.

    python indicator_grouping.py [path.csv]     # default: cohort/analysis_final.csv

Correlations are Pearson r on the 0/1 indicator vectors -- for binary vectors
that is the phi coefficient. Rows are dropped listwise on PRIMARY_OUTCOME first,
so the matrix is measured on exactly the rows the ensemble sees.

Writes reports/indicator_grouping.csv (one row per indicator pair, labelled
within-group / cross-group / ungrouped) and prints the per-group summary.
"""

import os
import sys

import pandas as pd

from tiers import INDICATOR_GROUPS, PRIMARY_OUTCOME

# Every value column datasets.py carries, including HIGH_MISSING_COLS, so the
# correlations cover every column that could get an indicator. (This used to
# import INDICATOR_ELIGIBLE_COLS, which datasets.py no longer defines.)
from datasets import INPUT_PATH, get_sensitivity_value_cols, load_data

INDICATOR_ELIGIBLE_COLS = get_sensitivity_value_cols()

DEFAULT_INPUT = INPUT_PATH
OUT_PATH = "reports/indicator_grouping.csv"

# A member whose indicator falls below this against its own panel would be
# carrying its own mechanism and should be split back out. Not enforced --
# printed as a flag, because the call is clinical, not numerical.
COHESION_FLOOR = 0.90


def indicator_frame(df, cols):
    """0/1 missingness indicators for cols, dropping never-missing columns."""
    ind = df[cols].isna().astype(float)
    return ind.loc[:, ind.std() > 0]


def groups_of(col, groups):
    """All groups containing col -- the panels deliberately overlap."""
    return [name for name, members in groups.items() if col in members]


def build_pairs(ind, groups):
    """One row per unordered indicator pair, with its group relationship."""
    corr = ind.corr()
    rows = []
    cols = list(ind.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            ga, gb = set(groups_of(a, groups)), set(groups_of(b, groups))
            shared = sorted(ga & gb)
            if shared:
                relation = "within " + ";".join(shared)
            elif not ga and not gb:
                relation = "ungrouped"
            else:
                relation = (f"cross {';'.join(sorted(ga)) or '-'} / "
                            f"{';'.join(sorted(gb)) or '-'}")
            rows.append({
                "col_a": a,
                "col_b": b,
                "r": round(float(corr.loc[a, b]), 4),
                "relation": relation,
            })
    return pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)


def within_group_r(ind, members):
    """The pairwise r values inside one group, as a flat list."""
    corr = ind[members].corr()
    return [float(corr.iloc[i, j])
            for i in range(len(members)) for j in range(i + 1, len(members))]


def summarize(ind, pairs, groups):
    """Per-group cohesion, plus each member's r against its own panel."""
    for name, members in groups.items():
        present = [c for c in members if c in ind.columns]
        missing = [c for c in members if c not in ind.columns]
        print(f"\n{name}  ({len(present)} members)")
        if missing:
            print(f"  not in this file / never missing: {missing}")
        if len(present) < 2:
            continue

        within = pd.Series(within_group_r(ind, present))
        panel = ind[present].max(axis=1)  # the collapsed indicator itself
        print(f"  panel missingness rate: {panel.mean():.2%}")
        print(f"  within-group pairwise r: mean {within.mean():.3f}, "
              f"min {within.min():.3f}, max {within.max():.3f}")

        print("  member r vs the collapsed panel indicator:")
        for col in present:
            r = float(ind[col].corr(panel))
            print(f"    {col:24s} miss={ind[col].mean():6.2%}  r={r:.3f}")

        # What was left out, and how close it came. This is the check that
        # decides whether a group is drawn too tightly or too loosely.
        outside = [c for c in ind.columns if c not in present]
        near = sorted(((float(ind[c].corr(panel)), c) for c in outside),
                      reverse=True)[:3]
        if near:
            print("  closest non-members:")
            for r, col in near:
                flag = "  <-- above cohesion floor" if r >= COHESION_FLOOR else ""
                print(f"    {col:24s} r={r:.3f}{flag}")


SHIPPED = "data/indicator30_grouped_simple.csv"


def verify_against_shipped(df):
    """Check the declared membership still reproduces the shipped variant.

    The grouped dataset was built by a script that no longer exists, so this is
    what keeps tiers.INDICATOR_GROUPS honest about what is actually in the file
    the ensemble was run on.
    """
    if not os.path.exists(SHIPPED):
        print(f"\n{SHIPPED} absent, membership not cross-checked")
        return
    shipped = pd.read_csv(SHIPPED)
    print(f"\nReproducing {SHIPPED}:")
    for name, members in INDICATOR_GROUPS.items():
        if name not in shipped.columns:
            continue
        if len(shipped) != len(df):
            print(f"  {name}: row counts differ, skipped")
            continue
        rebuilt = df[members].isna().any(axis=1).astype(float).values
        ok = bool((rebuilt == shipped[name].values).all())
        print(f"  {name}: {'exact match' if ok else 'MISMATCH'}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    if not os.path.exists(path):
        sys.exit(f"missing input: {path}")

    # datasets.load_data(), not a plain dropna on PRIMARY_OUTCOME -- see the
    # note in missingness_report.main(). The two differ by ~1,100 rows, and the
    # r values quoted in tiers.py have to come from the frame the ensemble
    # actually searches over.
    df = load_data(path)

    cols = [c for c in INDICATOR_ELIGIBLE_COLS if c in df.columns]
    ind = indicator_frame(df, cols)
    print(f"{len(ind.columns)} columns with any missingness, "
          f"{len(cols) - len(ind.columns)} never missing and dropped")

    pairs = build_pairs(ind, INDICATOR_GROUPS)
    summarize(ind, pairs, INDICATOR_GROUPS)

    # The separation that justifies keeping the panels apart rather than
    # collapsing everything onto one indicator.
    print("\nCross-group indicator correlation (mean |r| between members):")
    names = list(INDICATOR_GROUPS)
    corr = ind.corr()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma = [c for c in INDICATOR_GROUPS[a] if c in ind.columns]
            mb = [c for c in INDICATOR_GROUPS[b] if c in ind.columns]
            overlap = sorted(set(ma) & set(mb))
            block = corr.loc[ma, mb].abs()
            note = f"  (overlapping members: {len(overlap)})" if overlap else ""
            print(f"  {a} vs {b}: mean |r| {block.values.mean():.3f}{note}")

    print("\nHighest-r pairs overall:")
    print(pairs.head(10).to_string(index=False))

    verify_against_shipped(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    pairs.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
