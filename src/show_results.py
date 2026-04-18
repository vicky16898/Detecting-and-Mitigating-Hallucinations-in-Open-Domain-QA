import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default=None,
                    help="Strategy to show (default: all available)")
args = parser.parse_args()

results_root = "./data/helm/results"

if not os.path.exists(results_root):
    print(f"No results found at {results_root}")
    raise SystemExit(1)

strategies = (
    [args.strategy] if args.strategy
    else sorted(os.listdir(results_root))
)
strategies = [s for s in strategies if os.path.isdir(os.path.join(results_root, s))]

METRICS = [
    ("sent_halu.csv", "Sentence AUC (halu)", ".2f"),
    ("psg_halu.csv",  "Passage AUC (halu)",  ".2f"),
    ("sent_corr.csv", "Sentence Correlation", ".4f"),
    ("psg_corr.csv",  "Passage Correlation",  ".4f"),
]


def print_table(title, table):
    col_w = max(max(len(c) for c in table.columns), 10)
    row_w = max(max(len(r) for r in table.index), len("Strategy"))

    header = f"  {'Strategy':<{row_w}}  " + "  ".join(f"{c:>{col_w}}" for c in table.columns)
    sep    = f"  {'-' * row_w}  " + "  ".join("-" * col_w for _ in table.columns)

    print(f"\n  {title}")
    print(f"  {'=' * (len(header) - 2)}")
    print(header)
    print(sep)

    original_rows = [(s, row) for s, row in table.iterrows() if s == "original"]
    other_rows    = [(s, row) for s, row in table.iterrows() if s != "original"]

    for strategy, row in original_rows:
        print(f"  {strategy:<{row_w}}  " + "  ".join(f"{v:>{col_w}}" for v in row))
    if original_rows and other_rows:
        print(sep)
    for strategy, row in other_rows:
        print(f"  {strategy:<{row_w}}  " + "  ".join(f"{v:>{col_w}}" for v in row))
    print()


for fname, metric_label, fmt in METRICS:
    rows = {}
    for strategy in strategies:
        path = os.path.join(results_root, strategy, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        rows[strategy] = df["Our_score"]

    if not rows:
        continue

    table = pd.DataFrame(rows).T

    # Sort non-original rows by mean score descending before formatting
    original = table[table.index == "original"]
    others   = table[table.index != "original"].sort_values(
        by=table.columns.tolist(), ascending=False,
        key=lambda col: col.apply(pd.to_numeric, errors="coerce")
    )
    table = pd.concat([original, others])
    table = table.map(lambda x: format(x, fmt))
    print_table(metric_label, table)
