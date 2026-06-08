import argparse
import json
from pathlib import Path


def load_metrics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fmt_float(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamba", default="runs/mamba_metrics.json")
    parser.add_argument("--transformer", default="runs/transformer_metrics.json")
    args = parser.parse_args()

    rows = [load_metrics(args.mamba), load_metrics(args.transformer)]
    columns = [
        "model",
        "backend",
        "parameters",
        "best_val_acc",
        "test_acc",
        "train_seconds",
        "inference_images_per_second",
    ]

    widths = {
        col: max(len(col), *(len(fmt_float(row[col])) for row in rows))
        for col in columns
    }
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    divider = "-+-".join("-" * widths[col] for col in columns)
    print(header)
    print(divider)
    for row in rows:
        print(" | ".join(fmt_float(row[col]).ljust(widths[col]) for col in columns))


if __name__ == "__main__":
    main()
