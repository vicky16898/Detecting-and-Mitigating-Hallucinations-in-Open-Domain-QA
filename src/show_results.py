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
    for strategy, row in table.iterrows():
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
    table = table.map(lambda x: format(x, fmt))
    print_table(metric_label, table)
